#%%
import networkx as nx
from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch
import pickle
from whole_cell import WholeCell
from random_function import random_system, print_system, f_pow

#%%
num_nodes = 10
max_terms = 3
library = [f_pow(1), f_pow(0), torch.sin, torch.cos, f_pow(2)]
system, terms = random_system(num_nodes, library, krange=1/10, max_terms=3, self_deg=True)
print_system(terms)

graph = nx.DiGraph()
for i, eqn in enumerate(terms):
    for j, term in enumerate(eqn):
        if term is not None:
            graph.add_edge(j, i, term=term)    

pickle.dump(graph, open('test_graph.pkl','wb'))
nstarts = 3
starts = torch.rand(nstarts, num_nodes)
h = 0.1
maxsteps = 15000
eps = 1e-6
runs = []
for i, start in enumerate(starts):
    print(i)
    run = torch.zeros(maxsteps, num_nodes)
    x = start
    for j in range(maxsteps):
        # Euler step
        u = system(x)
        du = torch.max(torch.abs(h*u))
        if du < eps:
            print('converged')
            print(x)
            runs.append(run[:j])
            break
        x = x + h*u
        run[j] = x

# breakpoint()
# for run in runs: print(run[-1])
# from matplotlib import pyplot as plt
# for i in range(num_nodes):
#     plt.plot(run[:,i])
# plt.show()

#%%
#centers = torch.rand(3, graph.number_of_nodes())
#states = (centers + noise).view(-1, num_nodes)
#torch.save(states, 'test_states.torch')
type_mask = [torch.rand(1) > 0.5 for _ in range(num_nodes)]
torch.save(type_mask, 'test_type_mask.torch')

model = WholeCell(graph)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='sum')


# Start at a random position, with some nodes masked, run until fixed point 
# then compute loss against the nearest data point?

n_epoch = 5000
step_penalty = 1
                                            
def loss(states, type_mask):
    # Assign random values to the states of the masked nodes
    masked_state = states.clone()
    masked_state[:,type_mask] = torch.rand(len(states), sum(type_mask))
    trace, eps = model(masked_state, return_trace=True, return_eps=True)
    fixed_points = trace[-1]
    loss = loss_fn(fixed_points, states)
    
    return loss, eps

for i in range(n_epoch):
    optimizer.zero_grad()
    loss_mode_1, eps1 = loss(states, type_mask)
    step_len_loss_1 = step_penalty*torch.mean(eps1)
    # Invert the mask and assign random values 
    inverted_type_mask = [not mask for mask in type_mask]
    loss_mode_2, eps2 = loss(states, inverted_type_mask)
    step_len_loss_2 = step_penalty*torch.mean(eps2)

    total_loss = loss_mode_1 + loss_mode_2 # + step_len_loss_1 + step_len_loss_2
    total_loss.backward()

    print(i,' '.join([f'{x.item():.5f}' for x in 
         [total_loss, loss_mode_1, loss_mode_2, step_len_loss_1, step_len_loss_2]]))

    optimizer.step()

torch.save(model.state_dict(), 'simple_model.torch')
# breakpoint()
# %%
