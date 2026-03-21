import torch.nn as nn
import torch
import torch.nn.functional as F

"""
In this code we will work on the Encoder part the putput will be mean value of representations r for all input data points.
"""


class Encoder(nn.Module):
    """Processes (x, y) pairs into local representations r_i."""
    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        
        )

    def forward(self, x, y):
        # Input shape: [Batch, N, dim]
        combined = torch.cat([x, y], dim=-1)
        B, N, D = combined.shape
        h = self.net(combined.view(B*N, D))
        r_i = h.view(B,N,-1)
        return r_i


class Robust_Encoder(nn.Module):
    def __init__(self, x_size, y_size, hidden_dim):
        # Inherit the nn.Module class attributes
        super().__init__()
        self.input_size = x_size + y_size
        self.hidden_dim = hidden_dim
        
        # Deep MLP wiht LayerNorm for stability.
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
            
        )
        
    def forward(self, x, y):
        combined_input = torch.cat([x,y], dim = -1)
        r_i = self.net(combined_input)
        # aggregation 
        r = torch.mean(r_i, dim =1)
        return r 
            
        
    





"""
In next step we will design a function that will take this r values as an input and will give the parameters of the distribution as an output. Such as mean and variance. And from there we will sample the z values.

"""
