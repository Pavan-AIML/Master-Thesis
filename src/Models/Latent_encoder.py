import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

""" 
From here as we can see that the input to this module will be the output of the Encoder module which is r.
"""


class Latent_Encoder(nn.Module):
    """Maps aggregated r to the parameters of the latent distribution z."""
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Separate heads for mu and log_var are critical
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, r):
        hidden = self.pre_layer(r)
        mu = self.mu_head(hidden)
        log_sigma = self.log_sigma_head(hidden)
        # Clamp for numerical stability during training
        "Here we want the variance needs to remain closer to (0,1) as if the distribution is closer to normal distribution then handelling the noice will bbe easier as the function will not provide some random distribution."
        sigma =  0.1 + 0.9 * F.sigmoid(log_sigma)
        dist_z = Normal(loc = mu, scale= sigma)
        " Here dist will be the distribution of randomvariable z"
        return dist_z, mu, sigma




