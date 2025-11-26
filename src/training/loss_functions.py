"""
In this file we will design the loss functions for the training of the neural process model.

The loss function will consist of two parts:
1. Negative log likelihood loss between the reconstructed y (with mean and variance)values and the actual y values.


2. KL divergence loss between the latent distribution (with mean and variance) obtained from the context points and the latent distribution obtained from the target points.
"""

import sys
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)

class LossFunctions:
    def __init__(self, beta, learning_rate, stepsize, Number_of_steps, device):
        self.beta = beta
        self.learning_rate = learning_rate
        self.stepsize = stepsize
        self.Number_of_steps = Number_of_steps
        self.device = device
        
    def negative_log_likelihood(self, mu_y, var_y, y_target):
        # Calculating the negative log likelihood loss
        NLL = 
        
