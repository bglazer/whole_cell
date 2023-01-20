from torch.nn import Linear, Sequential, ReLU, LeakyReLU
import torch
from torchdyn.core import NeuralODE

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
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(WholeCell, self).__init__()
        self.model = MLP(input_dim, output_dim, hidden_dim, num_layers)
        self.neural_ode = NeuralODE(self.model, sensitivity='adjoint')  

    def forward(self, state):
        # state is a tensor of shape (num_nodes, num_states)
        delta = self.neural_ode(state)
        return delta

    def trajectory(self, state, tspan):
        # state is a tensor of shape (num_nodes, num_states)
        # tspan is a tensor of shape (num_timesteps,)
        trajectory = self.neural_ode.trajectory(state, tspan)
        return trajectory