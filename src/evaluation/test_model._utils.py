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

# os.path.dirname(__file__) >> Current file path we go one step above and then add the absolute path in the sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.optimizer_utils import context_target_split
from training.optimizer_utils import neural_process_data
from Models.neural_process import NeuralProcess
from configs.utils import config
from locationencoder.final_location_encoder import Geospatial_Encoder
from Dataloader.Modis_Data_loader.final_loader import Final_Air_Quality_Dataset_pipeline
from src.Models.neural_process import NeuralProcess
from training.optimizer_utils import neural_process_data, context_target_split


# Evaluator class
check_point_dir = "/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Master-Thesis/cpu_checkpoints/latitude_longitude____AOD/20260102-235714"

files = glob.glob(os.path.join(check_point_dir, "*.pth"))
files
latest_file = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
latest_file


class NP_Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    # TO evaluate the model we will first upload the best weights.

    def load_best_checkpoints(self, checkpoint_dir_path):
        files = glob.glob(os.path.join(checkpoint_dir_path, "*.pth"))
        if not files:
            raise FileNotFoundError(
                f"There are no check point files in the given {check_point_dir}"
            )

        try:
            latest_file = max(files, lambda t: int(t.split("_")[0].split(".")[0]))

        except ValueError:
            print(
                f" No better weights found in the check point directory and no {latest_file}"
            )

        # Now once we found the file of the best weights we will load them.

        pass
