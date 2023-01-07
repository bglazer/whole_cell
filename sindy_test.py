import numpy as np
import pysindy as ps
from whole_cell import WholeCell
import torch
import pickle

states = torch.load('test_states.torch')
type_mask = torch.load('test_type_mask.torch')
graph = pickle.load(open('test_graph.pkl','rb'))

model = WholeCell(graph)
model.load_state_dict(torch.load('simple_model.torch'))

num_nodes = graph.number_of_nodes()

for i in range(states.shape[0]):
    target = states[[i]]
    # Randomly sample points near the fixed points
    samples = torch.abs(target + torch.randn(num_nodes)/5)
    trace = model(samples, return_trace=True)
    print(len(trace))
    print(trace[-1][:,:5].detach().numpy())
    print(target[:,:5].detach().numpy())
    print('---')
    # breakpoint()

# t = np.linspace(0, 1, 100)
# x = 3 * np.exp(-2 * t)
# y = 0.5 * np.exp(t)
# X = np.stack((x, y), axis=-1)  # First column is x, second is y

# model = ps.SINDy(feature_names=["x", "y"])
# model.fit(X, t=t)
# model.print()