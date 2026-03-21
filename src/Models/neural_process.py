# Importing all the necessary libraries
import os
import sys
from pathlib import Path
import torch

# making the root path for the project
ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)
# importing all the necessary packages.

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Models.Encoder import Encoder
from src.Models.Latent_encoder import Latent_Encoder
from src.Models.Decoder import Decoder

# Import of the attentive neural process encoder, decoder, latent_encoder, spatial_encoder

from src.Models.Latent_encoder import Robust_Latent_Encoder
from src.Models.Encoder import Robust_Encoder
from src.Models.Decoder import Robust_Decoder

# to calculate the queries and keys
from src.Models.Encoder import Spatial_Encoder


"""
In this code we will be building the neural process network which will consist of Encoder, Latent Encoder and Decoder.

x_C_dim : dimension of context input x values
y_C_dim : dimension of context output y values
x_T_dim : dimension of target input x values
y_T_dim : dimension of target output y values
hidden_dim : dimension of hidden layers in all the modules
latent_dim : dimension of the latent variable z

"""

class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, latent_dim):
        super().__init__()
        self.Encoder = Encoder(x_dim, y_dim, hidden_dim)
        self.Latent_Encoder = Latent_Encoder(hidden_dim, latent_dim)
        self.Decoder = Decoder(x_dim, latent_dim, hidden_dim, y_dim)  

    def forward(self, x_context, y_context, x_target, y_target = None):
        r_context_i = self.Encoder(x_context, y_context)
        r_context_mean = torch.mean(r_context_i, dim =1)
        "For the pror distribution"

        dist_z_prior, mu_z_prior, sigma_z_prior = self.Latent_Encoder(r_context_mean)

        if y_target is not None:
            "to sample the global latent variable"
            x_all = torch.cat([x_context, x_target], dim =1)
            y_all = torch.cat([y_context, y_target], dim =1)
            r_all_i = self.Encoder(x_all, y_all)
            r_all_mean = torch.mean(r_all_i, dim =1)
            dist_z_posterior, mu_z_posterior, sigma_z_posterior = self.Latent_Encoder(r_all_mean)
            z = dist_z_posterior.rsample()
        else:
            dist_z_posterior, mu_z_posterior, sigma_z_posterior = dist_z_prior, mu_z_prior, sigma_z_prior
            z = dist_z_prior.rsample()

        dist_y, mu_y, sigma_y = self.Decoder(x_target, z)
        return {
            "dist_y": dist_y,
            "mu_y": mu_y,
            "sigma_y": sigma_y,
            "mu_z_prior": mu_z_prior,
            "sigma_z_prior": sigma_z_prior,
            "mu_z_posterior": mu_z_posterior,
            "sigma_z_posterior": sigma_z_posterior,
            "dist_z_prior": dist_z_prior,
            "dist_Z_posterior": dist_z_posterior
        }
        
    def predict(self, x_context, y_context, x_target):
        self.eval()
        with torch.no_grad():
            r_context_i = self.Encoder(x_context, y_context)
            r_context_mean = r_context_i.mean(dim=1)
            dist_z_prior, mu_z_prior, sigma_z_prior = self.Latent_Encoder(r_context_mean)
            z = dist_z_prior.rsample()
            dist_y, mu_y, sigma_y = self.Decoder(x_target, z)
            var_y = sigma_y ** 2
        return mu_y, var_y
        

"""
In this code we will be building the neural process network which will consist of Encoder, Latent Encoder and Decoder.

x_C_dim : dimension of context input x values
y_C_dim : dimension of context output y values
x_T_dim : dimension of target input x values
y_T_dim : dimension of target output y values
hidden_dim : dimension of hidden layers in all the modules
latent_dim : dimension of the latent variable z

"""




# Returning mu and log_var for the KL divergence calculation
# in this file mu_y, var_y are the reconstructed y values with uncertainty. To construct the loss function
# we will use the negative log likelihood loss between the reconstructed y values and the actual y values.
# And reconstructed y values are not deterministic but they are probabilistic with mean mu_y and variance var_y.
# mu and log_var are the parameters of the latent distribution. to create the KL divergence loss.

# EXAMLPE for learning purpose.

# r_c = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape = [2, 3]

# r_t = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])


# r_all = torch.stack([r_c, r_t], dim=1)
# r_all.mean(dim=1)
