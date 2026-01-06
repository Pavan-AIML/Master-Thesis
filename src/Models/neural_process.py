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
    def __init__(self, x_c_dim, y_c_dim, x_t_dim, y_t_dim, hidden_dim, latent_dim):
        super().__init__()
        self.x_c_dim = x_c_dim
        self.y_c_dim = y_c_dim
        self.x_t_dim = x_t_dim
        self.y_t_dim = y_t_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.x_c_dim, self.y_c_dim, self.hidden_dim)
        self.latent_encoder = Latent_Encoder(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(
            self.x_t_dim, self.latent_dim, self.hidden_dim, self.y_t_dim
        )

    def forward(self, x_context, y_context, x_target, y_target):
        # Step 1: Pass the context points through the encoder to get r
        rc = self.encoder(x_context, y_context)
        rt = self.encoder(x_target, y_target)
        # here we are
        rct = torch.stack([rc, rt], dim=1)
        rct = rct.mean(dim=1)
        # Step 2: Pass r through the latent encoder to get the parameters of the latent variable
        # Here we are predicting log variance. so that it confirms that it is a positive number and will be easier at the time of sampling.
        mu_zc, log_var_zc = self.latent_encoder(rc)
        # Sampling form global latent variable
        # And that is the reason we are having the input r from both context and targets.
        mu_zct, log_var_zct = self.latent_encoder(rct)
        # Step 3: Sample z from the latent distribution for contextual points
        stdct = torch.exp(0.5 * log_var_zct)
        epsct = torch.randn_like(stdct)
        zct = mu_zct + epsct * stdct
        # Step 4: Pass the target points and sampled z through the decoder to get the reconstructed y values
        mu_y, var_y = self.decoder(x_target, zct)
        # mu_y, var_y --> reconstruction loss
        # mu_zc, log_var_zc --> prior z distrubution
        # mu_zct, log_var_zct --> posterier z distribution
        return (mu_y, var_y, mu_zc, log_var_zc, mu_zct, log_var_zct)

    def predict(self, x_context, y_context, x_target):
        rc = self.encoder(x_context, y_context)
        mu, logvar = self.latent_encoder(rc)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        mu_yt, var_yt = self.decoder(x_target, z)
        # target samples.
        return (mu_yt, var_yt)


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
