# Helper utilities for evaluation (moved from test file to avoid pytest collection)
# This file contains helper classes and functions used by evaluation scripts.

# import all the necessary packages
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import torch.nn.functional as F

# The original test helper was incorrectly named with a leading 'test_' which
# caused pytest to try to collect it as a test; moving it here prevents that.


# (Content intentionally minimal — only a placeholder for utilities.)


def placeholder():
    """Placeholder utility to indicate this module is present."""
    return True
