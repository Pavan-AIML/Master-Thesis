import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.distributions import Independent
from torch.distributions import Normal

"""
In this module the input will be target values of x & z values and output values will be the reconstructed y values with uncertainty so here we will not get deterministic values of y but it's distirbution mu and variance.
"""


class Decoder(nn.Module):
    """Predicts target y given target x and sampled latent z."""
    def __init__(self, x_target_dim, z_dim, hidden_dim, y_target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_target_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_target_dim * 2), # Output mu and log_var
        )
        self.y_target_dim = y_target_dim

    def forward(self, x_target, z):
        # Expand z to match the number of target points: [B, 1, z_dim] -> [B, N_target, z_dim]
        B, N_t , _ = x_target.shape

        z = z.unsqueeze(1).expand(-1, N_t, -1)

        combined = torch.cat([x_target, z], dim=-1)
        h = self.net(combined.view(B*N_t, -1)).view(B, N_t, 2 * self.y_target_dim)

        mu, log_sigma = torch.chunk(h, 2, dim=-1)
        


       

        
        # Stability: Ensure variance is positive and doesn't collapse to zero
        # 0.1 is a minimum noise floor to prevent infinite likelihood
        " This will ensure the stability of the trainin of the tenworks so that the gradients are not dead."

        "Here the main reason of using the soft plus is that the noise should also be cpatured and noise can go beyond the range of 0 to 1. Hene not sigmoide."
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        dist = Normal(loc = mu, scale = sigma)
        base = Normal(loc= mu, scale = sigma)
        dost = Independent(base, 1)
        return dist, mu, sigma


"""
From here the outputs will be the mean and variance of the reconstructed y values.
Now we will work on building the neural process network.
"""
