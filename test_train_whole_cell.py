#%%
import torch
from scipy.spatial import KDTree
from whole_cell import WholeCell
from random_function import random_system, print_system, f_pow
from matplotlib import pyplot as plt

#%%
num_nodes = 2
max_terms = 3
library = [f_pow(1), f_pow(0), torch.sin, torch.cos, f_pow(2)]
system, terms = random_system(num_nodes, library, krange=1/10, max_terms=3, self_deg=True)
print_system(terms)

#%%
nstarts = 15
starts = torch.randn(nstarts, num_nodes)*2
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
            print('converged', flush=True)
            print(x)
            runs.append(run[:j])
            break
        x = x + h*u
        run[j] = x

#%%
data = torch.cat(runs)
sample_factor = 10
# Take every Nth point from each run
sampled = torch.zeros(0, num_nodes)
for run in runs:
    sampled = torch.cat([sampled, run[::sample_factor]])
#for run in runs: print(run[-1])
from matplotlib import pyplot as plt
for i in range(nstarts):
    plt.scatter(runs[i][:,0], runs[i][:,1], s=1)
# TODO Add noise to the sampled points
# TODO maybe add noise during the simulation
noise_factor = 0.1
noisy = sampled + torch.randn(sampled.shape)*noise_factor

#%%
#plt.scatter(sampled[:,0], sampled[:,1], s=1)


#%% 
# Compute pseudo-time trajectories using scanpy
from scanpy.tl import paga, leiden, dpt, diffmap
from scanpy.pp import neighbors
from scanpy import AnnData
import scanpy as sc
import numpy as np
adata = AnnData(sampled.detach().numpy())
print('Computing neighbors')
n_neighbors = 10
neighbors(adata, n_neighbors=n_neighbors)
print('Computing leiden clusters')
leiden(adata)
print('Computing PAGA')
paga(adata)
print('Setting root')
end = runs[0][-1].detach().numpy()
adata.uns['iroot'] = np.flatnonzero(np.abs(adata.X - end)<1e-5)[0]
print('Computing diffmap')
diffmap(adata)
print('Computing diffusion pseudotime')
dpt(adata)


#%%
# Get the nearest neighbors that have a larger pseudo-time for each point
transition_points = []
for i in range(len(sampled)):
    # Sparse matrix row for the current point
    row=adata.obsp['distances'][i]
    # indexes of non-zero entries in the sparse matrix row
    nonzeros = row.nonzero()[1]
    neighbor_pseudotimes = adata.obs['dpt_pseudotime'][nonzeros]
    current_pseudotime = adata.obs['dpt_pseudotime'][i]
    larger_pseudotimes = neighbor_pseudotimes < current_pseudotime
    # Check if both points are in the same cluster
    same_cluster = adata.obs['leiden'][i] == adata.obs['leiden'][nonzeros[larger_pseudotimes]]
    # Check if their clusters are connected in the paga graph
    # TODO TODO TODO
    idxs = np.flatnonzero(same_cluster & larger_pseudotimes)

    if sum(idxs) == 0:
        transition_points.append(sampled[i])
    else:
        next_points = nonzeros[idxs]
        mean_transition_point = torch.mean(sampled[next_points], dim=0)
        min_dist_nonzero_idx = adata.obsp['distances'][i, next_points].argmin()
        min_dist_idx = next_points[min_dist_nonzero_idx]
        min_dist_point = sampled[min_dist_idx]
        transition_points.append(min_dist_point)

min_transition_points = torch.vstack(transition_points)
# mean_transition_points = torch.vstack([torch.mean(sampled[points], dim=0) for points in transition_points])

#%%
def plot_arrows(start_points, next_points, sample_factor=10, save_file=None):
    # Plot the vectors from the sampled points to the transition points
    d = next_points - start_points
    # Increase the size of the plot to see the vectors
    plt.figure(figsize=(15,15))
    # Plot the data with the pseudo-time as color
    # TODO: depends on global context
    #plt.scatter(adata.X[:,0], adata.X[:,1], c=adata.obs['dpt_pseudotime'], s=5)
    
    for i in range(0, len(d), sample_factor):
        plt.arrow(start_points[i,0], start_points[i,1], d[i,0], d[i,1], color='r', alpha=1)

    if save_file is not None:
        plt.savefig(save_file)
        plt.close()

plot_arrows(sampled.detach().cpu(), min_transition_points.detach().cpu())

#%%
device = 'cuda:0'
model = WholeCell(input_dim=num_nodes, 
                  output_dim=num_nodes, 
                  hidden_dim=12, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(reduction='mean')
sampled = sampled.to(device)
min_transition_points = min_transition_points.to(device)
#%%
mse = torch.nn.MSELoss(reduction='mean')

n_epoch = 10000
n_points = 1000
n_traces = 50
n_samples = 10
  
#%%
for i in range(n_epoch):
    optimizer.zero_grad()
    # Run the model from N randomly selected data points
    # Random sampling
    # idxs = torch.randint(0, sampled.shape[0], (n_points,))
    # Full set of points
    idxs = torch.arange(sampled.shape[0])
    starts = sampled[idxs]
    _, fx = model(starts, tspan=torch.linspace(0,1,2))
    # fx = model.model(starts)
    min_next_points = min_transition_points[idxs]
    # TODO mmd doesn't really make sense if we don't have stochasticity in the model
    # mse might not make sense either
    loss = 1000*mse(fx[-1], min_next_points)
    # loss = mse(fx, min_next_points)
    loss.backward()
    optimizer.step()
    print(i,' '.join([f'{x.item():.9f}' for x in 
          [loss]]), flush=True)
    if i % 100 == 0:
        plot_arrows(sampled.detach().cpu(), 
                    fx[-1].detach().cpu(),
                    sample_factor=10,
                    save_file=f'figures/test/vector_field_{i}.png')
    if i%1000 == 0:
        plot_traces(model, sampled, i, n_traces=100)

#%%
def plot_traces(model, sampled, i, n_traces=50):
    # Generate some sample traces
    trace_starts = sampled[torch.randint(0, sampled.shape[0], (n_traces,))]
    traces = model.trajectory(state=trace_starts, tspan=torch.linspace(0, 100, 500))
    trace_plot = traces.cpu().detach().numpy()
    # Create a new figure
    fig, ax = plt.subplots()
    # Plot the data with partial transparency so that we can highlight the traces
    ax.scatter(data[:,0], data[:,1], s=.25, alpha=0.1)
    for trace in range(n_traces):
        # Plot the traces
        ax.scatter(trace_plot[:,trace,0], trace_plot[:,trace,1], s=1)
    # Save the plot to a file indicating the epoch
    plt.savefig(f'figures/test/traces_{i}.png')
    plt.close()



torch.save(model.state_dict(), 'simple_model.torch')
# %%
