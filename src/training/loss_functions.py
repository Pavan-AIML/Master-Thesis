"""
In this file we will design the loss functions for the training of the neural process model.

The loss function will consist of two parts:
1. Negative log likelihood loss between the reconstructed y (with mean and variance)values and the actual y values.


2. KL divergence loss between the latent distribution (with mean and variance) obtained from the context points and the latent distribution obtained from the target points.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import os

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)


class LossFunctions(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # weight on KL divergence term

    # ----------- Negative log likely hood ------------------------------------

    def negative_log_likelihood(self, mu_y, var_y, y_target):
        # Calculating the negative log likelihood loss it will be between mean_y and variance_y wit the help of negative log likelihood function.
        eps = 1e-6
        var_y = var_y + eps  # To have the stability in variance

        NLL = 0.5 * (
            (torch.log(2 * torch.pi * var_y)) + ((y_target - mu_y) ** 2 / var_y)
        )
        NLL = NLL.sum(dim=[1, 2]).mean()
        return NLL

    # -------------------- KL Divergence Term ------------------------

    # now we will define the KL divergence loos to minimize the distribution between them.
    # mu_zc ---> mean of z by just using contextual points
    # mu_zct ---> mean of z by usieng contextual and target points
    # log_var_zc ---> log variance of z with onlz contextual points
    # log_var_zct ---> log variance of z with contextual and target points

    #
    def KL_divergence(self, mu_zc, log_var_zc, mu_zct, log_var_zct):
        # to avoide the variance going below the expectation we will clamp it to avoide the failure of the training.
        log_var_zc = torch.clamp(log_var_zc, -10, 10)
        log_var_zct = torch.clamp(log_var_zct, -10, 10)
        var_zct = log_var_zct.exp()
        var_zc = log_var_zc.exp()

        KL = -0.5 * (
            (log_var_zc - log_var_zct)
            + ((var_zct + (mu_zct - mu_zc) ** 2) / var_zc)
            - 1
        )
        return KL.sum(dim=1).mean()

    """
    here we will compute the full neural process loss. 
    """

    def forward(self, mu_y, var_y, y_target, mu_zc, mu_zct, log_var_zc, log_var_zct):
        nll = self.negative_log_likelihood(mu_y, var_y, y_target)
        kl = self.KL_divergence(mu_zc, log_var_zc, mu_zct, log_var_zct)
        total_loss = nll + kl
        return total_loss, nll, kl
