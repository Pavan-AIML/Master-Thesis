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

"""
Validation data set 
"""


from notebooks.Modis_data_analysis.final_loader import (
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25_val,
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_val,
    Final_Air_Quality_Dataset_pipeline_latlon_PM25_val,
    Final_Air_Quality_Dataset_pipeline_latlon_val,
)


"""
Test data set 
"""
from notebooks.Modis_data_analysis.final_loader import (
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25_test,
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_test,
    Final_Air_Quality_Dataset_pipeline_latlon_PM25_test,
    Final_Air_Quality_Dataset_pipeline_latlon_test,
)

# importing the loss function
from loss_functions import LossFunctions


"""
Instances for the training data set 
"""
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

final_data_latlong_AOD_PM25[1].shape
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
x.shape
final_data_latlong_PM25
final_data_latlong


"""
Instances for the test data 

"""
instance_latlon_AOD_PM25_val = Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25_val(
    config
)
instance_latlon_AOD_val = Final_Air_Quality_Dataset_pipeline_latlon_AOD_val(config)
instance_latlon_PM25_val = Final_Air_Quality_Dataset_pipeline_latlon_PM25_val(config)
instance_latlon_val = Final_Air_Quality_Dataset_pipeline_latlon_val(config)

"""
Final test data sets 
"""
# latlon_AOD_pm25

instance_latlon_AOD_PM25_val.modis_data_sets()
instance_latlon_AOD_PM25_val.stations_data_sets()
instance_latlon_AOD_PM25_val.PM_25_data()
instance_latlon_AOD_PM25_val.training_data()
instance_latlon_AOD_PM25_val.Torch_data()
final_data_latlong_AOD_PM25_val = instance_latlon_AOD_PM25_val.full_pipeline()
final_data_latlong_AOD_PM25_val

# latlon_AOD
instance_latlon_AOD_val.modis_data_sets()
instance_latlon_AOD_val.stations_data_sets()
instance_latlon_AOD_val.PM_25_data()
instance_latlon_AOD_val.training_data()
instance_latlon_AOD_val.Torch_data()
final_data_latlong_AOD_val = instance_latlon_AOD_PM25_val.full_pipeline()
final_data_latlong_AOD_val[0].shape


# latlon_PM2.5
instance_latlon_PM25_val.modis_data_sets()
instance_latlon_PM25_val.stations_data_sets()
instance_latlon_PM25_val.PM_25_data()
instance_latlon_PM25_val.training_data()
instance_latlon_PM25_val.Torch_data()
final_data_latlong_PM25_val = instance_latlon_AOD_PM25_val.full_pipeline()
final_data_latlong_PM25_val[0].shape


# latlon
instance_latlon_val.modis_data_sets()
instance_latlon_val.stations_data_sets()
instance_latlon_val.PM_25_data()
instance_latlon_val.training_data()
instance_latlon_val.Torch_data()
final_data_latlong_val = instance_latlon_AOD_PM25_val.full_pipeline()
final_data_latlong_val[0].shape

# ************************ Final validation data ***************************************
final_data_latlong_AOD_PM25_val
final_data_latlong_AOD_val
final_data_latlong_PM25_val
final_data_latlong_val
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


"""
neural_process_data converts the data from the final paipe line to dataloader format so that we can convert the data and access in the __getitem__ form.
"""

from optimizer_utils import neural_process_data
# now to start training we need a data-loader that

# for train data

NeuralProcessData_latlon_AOD_PM25 = neural_process_data(
    final_data_latlong_AOD_PM25[0],
    final_data_latlong_AOD_PM25[1],
    num_points_per_task=200,
)

# for test data

NeuralProcessData_latlon_AOD_PM25_val = neural_process_data(
    final_data_latlong_AOD_PM25_val[0],
    final_data_latlong_AOD_PM25_val[1],
    num_points_per_task=200,
)


# ------------------------------- practice --------------------

X_task, Y_task = NeuralProcessData_latlon_AOD_PM25[0]
X_task.shape
Y_task.shape

# ------------------------------- practice ----------------------------------
"""
Creating the final data loaders for train val and test sets.
"""

# ------------------- Here we will start training ---------------------------

# creating data loader for training data
dataloader = DataLoader(
    dataset=NeuralProcessData_latlon_AOD_PM25,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
dataloader.dataset

# checking the size and shape of data loader
for x, y in dataloader:
    print(x.shape, y.shape)
len(dataloader)


# creating data loader fo rvalidation data
dataloader_val = DataLoader(
    dataset=NeuralProcessData_latlon_AOD_PM25_val,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

# checking the size and shape of data loader
for x, y in dataloader_val:
    print(x.shape, y.shape)
len(dataloader_val)

# In this way we can extract the x, y batches from dataloader.

"""
How to check the training dataloader 
"""

len(dataloader)
Xb, Yb = next(iter(dataloader))
Xb.shape, Yb.shape


"""
Start training 
"""

# importing the trainer
from optimizer_utils import NPTrainer
from optimizer_utils import validation_function

# cretaing and saving the logs as per the current training.
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
unique_log_dir = f"./logs/latlon_AOD_PM25/{current_time}"

# Training the model

Training = NPTrainer(
    model=model_latlon_AOD_PM25,
    optimizer=Optimizer,
    loss_fn=Loss,
    device="cpu",
    log_dir=unique_log_dir,
    checkpoint_dir="./checkpoints/latlon_AOD_PM25",
)

# running the epochs.
# ... definitions above ...

# Define a "best loss" to track improvement
best_val_loss = float('inf')

for epoch in range(100): # Run for 100 epochs
    
    # 1. TRAIN
    train_loss = Training.train_epoch(dataloader, epoch)
    
    # 2. VALIDATE (Call the function we just wrote)
    val_loss = validation_function(
        model=model_latlon_AOD_PM25, # Access model from your Trainer class
        val_dataloader=dataloader_val, # You need a separate loader for val data
        loss_fn=Loss,
        device="cpu"
    )

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 3. SAVE ONLY IF BETTER
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Training.save_checkpoint(epoch)
        print(f"   >>> SAVED: New best model found!")


"""
Now we need to export the validation data-set so that our trained models can be validated and model with best weights can be selected.
"""

from optimizer_utils import Trained_model_selection_in_val_data_set

Model_selection_latlon_AOD_PM25 = Trained_model_selection_in_val_data_set(
    model_latlon_AOD_PM25, dataloader_val, "cpu", Loss
)



# self, model, val_dataloader, device, Loss, context_target_split
# Now we are extracting the weights and will be using it for the testing purpose we will make our model ready for test.
