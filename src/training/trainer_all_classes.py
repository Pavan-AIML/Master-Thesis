# Here we will create the dat aloader and train test split and trained and evaluator.
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split

x_c = torch.tensor([[1, 2, 3], [4, 5, 6]])
y_c = torch.tensor([[7, 8, 9], [10, 11, 12]])
x_t = torch.tensor([[13, 14, 15], [14, 15, 16]])
y_t = torch.tensor([[17, 18, 19], [20, 21, 22]])

"""
From this below class we will split the data sets in to context and targets.
"""


class NeuralProcessDataset(Dataset):
    """
    Creates context and target sets on the fly for Neural Processes.
    """

    def __init__(
        self, X, Y, min_context=10, max_context=50, min_target=50, max_target=100
    ):
        self.X = X  # Shape: [N, x_dim]
        self.Y = Y  # Shape: [N, y_dim]
        self.N = X.shape[0]

        self.min_context = min_context
        self.max_context = max_context
        self.min_target = min_target
        self.max_target = max_target

    def __len__(self):
        # Number of NP tasks per epoch
        return 3000

    def __getitem__(self, idx):
        perm = torch.randperm(self.N)

        # Sample random sizes
        C = torch.randint(self.min_context, self.max_context + 1, (1,)).item()
        T = torch.randint(self.min_target, self.max_target + 1, (1,)).item()

        # Select context
        ctx_idx = perm[:C]
        x_c = self.X[ctx_idx]  # [C, x_dim]
        y_c = self.Y[ctx_idx]  # [C, y_dim]

        # Select target
        tgt_idx = perm[C : C + T]
        x_t = self.X[tgt_idx]  # [T, x_dim]
        y_t = self.Y[tgt_idx]  # [T, y_dim]

        return x_c, y_c, x_t, y_t
