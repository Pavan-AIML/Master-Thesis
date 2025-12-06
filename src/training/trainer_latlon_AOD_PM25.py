"""
Importing the necessary packages.............
"""

import os
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import arrow
from datetime import datetime

# from utils.config import config
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)

from notebooks.Modis_data_analysis.utils import config

# -------------************---------------------------------
# loading all the data loaders here we will load the final data in torch.
from notebooks.Modis_data_analysis.PM25_data_loader_analysis import Modis_data_loader
from notebooks.Modis_data_analysis.PM25_data_loader_analysis import PM_25_dataloader
from notebooks.Modis_data_analysis.PM25_data_loader_analysis import (
    combine_the_data_frames,
)
from notebooks.Modis_data_analysis.PM25_data_loader_analysis import Training_data_loader
from locationencoder.final_location_encoder import Geospatial_Encoder
from notebooks.Modis_data_analysis.torch_data_loader import (
    AirQualityDataset_latlon_AOD_PM25,
    AirQualityDataset_latlon_AOD,
    AirQualityDataset_latlon_PM25,
    AirQualityDataset_latlon,
)

# importing config files
from notebooks.Modis_data_analysis.final_loader import (
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25,
    Final_Air_Quality_Dataset_pipeline_latlon_AOD,
    Final_Air_Quality_Dataset_pipeline_latlon_PM25,
    Final_Air_Quality_Dataset_pipeline_latlon,
)

# importing the loss function
from loss_functions import LossFunctions


# -------------************--------------------------------
# Creating all the instance here

instance_latlon_AOD_PM25 = Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25(config)
instance_latlon_AOD = Final_Air_Quality_Dataset_pipeline_latlon_AOD(config)
instance_latlon_PM25 = Final_Air_Quality_Dataset_pipeline_latlon_PM25(config)
intance_latlon = Final_Air_Quality_Dataset_pipeline_latlon(config)


# final data with latitude, longitude, AOD and PM2.5
instance_latlon_AOD_PM25.modis_data_sets()
instance_latlon_AOD_PM25.stations_data_sets()
instance_latlon_AOD_PM25.PM_25_data()
instance_latlon_AOD_PM25.training_data()
instance_latlon_AOD_PM25.Torch_data()
final_data_latlong_AOD_PM25 = instance_latlon_AOD_PM25.full_pipeline()
len(final_data_latlong_AOD_PM25[0])

final_data_latlong_AOD_PM25[1]
# final data with latitude, longitude and AOD

instance_latlon_AOD.modis_data_sets()
instance_latlon_AOD.stations_data_sets()
instance_latlon_AOD.PM_25_data()
instance_latlon_AOD.training_data()
instance_latlon_AOD.Torch_data()
final_data_latlong_AOD = instance_latlon_AOD.full_pipeline()
final_data_latlong_AOD[0]
final_data_latlong_AOD[1][0]

# final data with latitude, longitude  and PM2.5
instance_latlon_PM25.modis_data_sets()
instance_latlon_PM25.stations_data_sets()
instance_latlon_PM25.PM_25_data()
instance_latlon_PM25.training_data()
instance_latlon_PM25.Torch_data()
final_data_latlong_PM25 = instance_latlon_PM25.full_pipeline()
final_data_latlong_PM25[0].shape
final_data_latlong_PM25[1]

# final data with latitude and longitude
intance_latlon.modis_data_sets()
intance_latlon.stations_data_sets()
intance_latlon.PM_25_data()
intance_latlon.training_data()
intance_latlon.Torch_data()
final_data_latlong = intance_latlon.full_pipeline()
final_data_latlong[0].shape
final_data_latlong[1][0]


# -------------************--------------------------------
"""
Final training data.................
"""
final_data_latlong_AOD_PM25

final_data_latlong_AOD_PM25
final_data_latlong_AOD[0].shape
final_data_latlong_PM25[0].shape
final_data_latlong[0].shape

for i, t in enumerate(final_data_latlong_AOD_PM25):
    print(i, t.shape)
# final data sets wit x, y in torch tensor form
final_data_latlong_AOD_PM25[0].shape
x, y = final_data_latlong_AOD
x
final_data_latlong_PM25
final_data_latlong


# -------------************--------------------------------

# training the model in different sdata sets and storing the weights.

# -------------************--------------------------------

# importing the model from model file.
"""
Importing the model................
"""

from src.Models.neural_process import NeuralProcess

# self, x_c_dim, y_c_dim, x_t_dim, y_t_dim, hidden_dim, latent_dim
# self, x_target_dim, z_dim, hidden_dim, y_target_dim
# self, x_c_dim, y_c_dim, x_t_dim, y_t_dim, hidden_dim, latent_dim

# Creating the models those are compatible with the all kinds of inputs.

# -------------************---------------------------------

"""
Final models for training........... making attributes from the class
"""

model_latlon_AOD_PM25 = NeuralProcess(128, 2, 128, 2, 128, 128)
model_latlon_AOD = NeuralProcess(127, 2, 127, 2, 128, 128)
model_latlon_PM25 = NeuralProcess(127, 2, 127, 2, 128, 128)
model_latlon = NeuralProcess(126, 2, 127, 2, 128, 128)


"""
Here we will import the loss function...........
"""
# self, beta, learning_rate, stepsize, Number_of_steps, device#
Loss = LossFunctions()


"""
Now we have model, loss function, data from here we can build the model trainer that will train the model with the tensor board.

The first & second class --: will devide the data in to chunks and also train, val and test this will create a final data loader. 

Third class --: this class will create the loop for train the network. 

Fourth class --: will validate the data set 

Fifth step --: will evaluate the model.

"""

# ------------ The first step is to devide the data in terms of context and test sets.


from optimizer_utils import context_target_split

C_T_data = context_target_split(
    final_data_latlong_AOD_PM25[0].unsqueeze(0),
    final_data_latlong_AOD_PM25[1].unsqueeze(0),
)

C_T_data[0].shape
C_T_data[1].shape
C_T_data[2].shape
C_T_data[3].shape

final_data_latlong_AOD_PM25[0].unsqueeze(0)[0]

""" self,
        model,
        optimizer,
        loss_fn,
        device,
        log_dir="./logs",
        checkpoint_dir="./checkpoints",
"""

# Importing the optimizer
import torch.optim as optim

Optimizer = optim.Adam(model_latlon_AOD_PM25.parameters(), lr=1e-5)

# Importing the trainer

final_data_latlong_AOD_PM25[0].__getitem__(10)
# First data ser priority will be
# def train_epoch(self, dataloader, epoch_idx):

from optimizer_utils import neural_process_data
# now to start training we need a data-loader that

NeuralProcessData_latlon_AOD_PM25 = neural_process_data(
    final_data_latlong_AOD_PM25[0],
    final_data_latlong_AOD_PM25[1],
    num_points_per_task=200,
)

X_task, Y_task = NeuralProcessData_latlon_AOD_PM25[0]
X_task.shape
Y_task.shape

# ------------------- Here we will start training -----------------------------------------

dataloader = DataLoader(
    dataset=NeuralProcessData_latlon_AOD_PM25,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
dataloader.dataset


Xb, Yb = next(iter(dataloader))
Xb.shape, Yb.shape
from optimizer_utils import NPTrainer

# Training the model

Training = NPTrainer(
    model=model_latlon_AOD_PM25,
    optimizer=Optimizer,
    loss_fn=Loss,
    device="cpu",
    log_dir="./logs",
    checkpoint_dir="./checkpoints",
)

# running the epochs.
for epoch in range(10):
    Training.train_epoch(dataloader, epoch)
# saving the models weights.
for epoch in range(10):
    Training.save_checkpoint(epoch)

# Now we are extracting the weights and will be using it for the testing purpose.
