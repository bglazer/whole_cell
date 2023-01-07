import networkx as nx
from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch
import pickle
from whole_cell import WholeCell
from random_function import random_function

num_nodes = 10
graph = nx.erdos_renyi_graph(num_nodes, 0.2, directed=True)
pickle.dump(graph, open('test_graph.pkl','wb'))
model = WholeCell(graph)
noise = torch.rand(100, 3, graph.number_of_nodes())
centers = torch.rand(3, graph.number_of_nodes())
states = (centers + noise).view(-1, num_nodes)
torch.save(states, 'test_states.torch')
type_mask = [torch.rand(1) > 0.5 for _ in range(num_nodes)]
torch.save(type_mask, 'test_type_mask.torch')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='sum')

n_epoch = 5000
step_penalty = 1

fs = [random_function() for _ in range(num_nodes)]


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