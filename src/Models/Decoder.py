import torch
from torch.nn import nn
import torch.functionals as F

"""
In this modeule theh input will be target values of x & z values and output values will be the reconstructed y values with uncertainty so here we will not get deterministic values of y but it's distirbution mu and variance.
"""


class Decoder(nn.Module):
    def __init__(self, x_target_dim, z_dim, hidden_dim, y_target_dim):
        super().__init__()
        self.x_target_dim = x_target_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.y_target_dim = y_target_dim
        self.net = nn.Sequential(
            nn.Linear(self.x_target_dim + self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.y_target_dim * 2)  # Outputting both mean and log variance
        )
        
    def forward(self, x_target, z):
        z = z.expand(-1, x_target.size(1), z.size(-1))  # Expanding z to match x_target's shape
        combined_input = torch.cat([x_target, z], dim=-1)
        output = self.net(combined_input)
        mu, log_var = output.chunk(2, dim=-1)  # Splitting into mean and log variance
        var = torch.exp(2 * log_var)  # Ensuring variance is positive
        return mu, var

"""
From here the outputs will be the mean and variance of the reconstructed y values.
Now we will work on building the neural process network.
"""

