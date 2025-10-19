import torch
from torch.nn import nn
import torch.functionals as F

""" 
From here as we can see that the input to this module will be the output of the Encoder module which is r.
"""


class Latent_Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, r):
        mu = self.mu(r)
        log_var = self.log_var(r)
        return mu, log_var
