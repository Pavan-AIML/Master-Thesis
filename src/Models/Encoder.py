import torch.nn as nn
import torch
import torch.nn.functional as F

"""
In this code we will work on the Encoder part the putput will be mean value of representations r for all input data points.
"""

class Encoder(nn.Module):
    def __init__(self, x_size, y_size, hidden_size):
        super().__init__()
        self.input_size = x_size + y_size
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    # in this forward function we will convert our input to the lower dimension embeddings.
    def forward(self, x, y):
        # Here we will concatinate the input as well as output to create a combined input.
        combined_input = torch.cat([x, y], dim=-1)
        # Now we will pass this combined input through a linear layer to reduce its dimension.
        r_i = self.net(combined_input)
        r = torch.mean(r_i, dim=1)
        return r


"""
In next step we will design a function that will take this r values as an input and will give the parameters of the distribution as an output. Such as mean and variance. And from there we will sample the z values.

"""