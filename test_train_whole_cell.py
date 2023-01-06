import networkx as nx
from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch
import pickle
from whole_cell import WholeCell

num_nodes = 10
graph = nx.erdos_renyi_graph(num_nodes, 0.2, directed=True)
pickle.dump(graph, open('test_graph.pkl','wb'))
model = WholeCell(graph)
states = torch.rand(3, graph.number_of_nodes())
torch.save(states, 'test_states.torch')
type_mask = [torch.rand(1) > 0.5 for _ in range(num_nodes)]
torch.save(type_mask, 'test_type_mask.torch')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='sum')

n_epoch = 5000

for i in range(n_epoch):
    optimizer.zero_grad()

    # Assign random values to the states of the masked nodes
    masked_state = states.clone()
    masked_state[:,type_mask] = torch.rand(len(states), sum(type_mask))
    fixed_points = model(masked_state)
    loss = loss_fn(fixed_points, states)
    loss.backward(retain_graph=True)

    # Invert the mask and assign random values 
    type_mask = [not mask for mask in type_mask]
    masked_state = states.clone()
    masked_state[:,type_mask] = torch.rand(len(states), sum(type_mask))
    fixed_points = model(masked_state)

    loss += loss_fn(fixed_points, states)
    loss.backward()
    print(i,loss.item())

    optimizer.step()

torch.save(model.state_dict(), 'simple_model.torch')
breakpoint()