import networkx as nx
from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch

# This is a whole cell model in PyTorch. 
# The model is a directed graph where each node is a MLP. 
# The input to each node is the state of the nodes that are connected to it. 
# The output of each node is the state of the node itself. The state of each node represents the gene/protein expression at that node. 
# The model is supposed to converge to a fixed point

def MLP(input_dim, output_dim, hidden_dim, num_layers):
    layers = []
    layers.append(Linear(input_dim, hidden_dim))
    layers.append(LeakyReLU())
    for i in range(num_layers - 1):
        layers.append(Linear(hidden_dim, hidden_dim))
        layers.append(LeakyReLU())
        # TODO do we need batch norm here?
    layers.append(Linear(hidden_dim, output_dim, bias=False))
    layers.append(LeakyReLU())
    return Sequential(*layers)

class WholeCell(torch.nn.Module):
    def __init__(self, graph):
        super(WholeCell, self).__init__()
        self.graph = graph
        self.mlps = {}
        self_loops = [(node, node) for node in graph.nodes]
        self.graph.add_edges_from(self_loops)
        self.node_mapping = {node: i for i, node in enumerate(graph.nodes)}
        self.index_mapping = {i: node for i, node in enumerate(graph.nodes)}
        for node in self.graph:
            input_dim = self.graph.in_degree(node)
            self.mlps[node] = MLP(input_dim=input_dim, output_dim=1, 
                                  hidden_dim=100, num_layers=2)
            self.add_module(f'node-{node}', self.mlps[node])
        

    # TODO implement accelerated convergence method
    def forward(self, state, threshold=1e-6, return_trace=False, return_eps=False):
        eps = float('inf')
        new_state = torch.zeros_like(state)
        eps_trace = []
        if return_trace:
            trace = [state]
        while threshold < eps:
            for node, mlp in self.mlps.items():
                # Tensor of states of inputs to the node
                inputs = list(self.graph.predecessors(node))
                input_tensor = state[:,inputs]
                node_idx = self.node_mapping[node]
                new_state[:,node_idx] = mlp(input_tensor).squeeze()
            if trace:
                trace.append(new_state)
            eps = torch.linalg.norm(new_state - state)
            eps_trace.append(eps)
            # breakpoint()
            state = new_state

        # convert trace from a list of tensors to a tensor
        trace = torch.stack(trace)
        # convert eps_trace from a list of tensors to a tensor
        eps_trace = torch.stack(eps_trace)

        if return_trace:
            if return_eps:
                return trace, eps_trace
            return trace
        if return_eps:
            return new_state, eps_trace
        return state

