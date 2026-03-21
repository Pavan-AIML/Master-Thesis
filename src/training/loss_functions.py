from torch.distributions import kl_divergence
import sys
from pathlib import Path
import torch
import torch.nn as nn
import os

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)
from locationencoder.final_location_encoder import Geospatial_Encoder

class NPELBO(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, outputs, y_target):
        dist_y = outputs["dist_y"]
        dist_z_prior = outputs["dist_z_prior"]
        dist_z_posterior = outputs["dist_Z_posterior"]
        nll = -dist_y.log_prob(y_target).mean()
        kl_loss = kl_divergence(dist_z_posterior, dist_z_prior).sum(-1).mean()
        total_loss = nll + self.beta*kl_loss    
        return total_loss, nll, kl_loss

        