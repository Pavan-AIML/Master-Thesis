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
from torch import optim

# from utils.config import config
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)

from configs.utils import config

# -------------************---------------------------------
# loading all the data loaders here we will load the final data in torch.
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import Modis_data_loader
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import PM_25_dataloader
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import (
    combine_the_data_frames,
)
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import (
    Training_data_loader,
)
from locationencoder.final_location_encoder import Geospatial_Encoder

geospatial_encoder = Geospatial_Encoder(
    config["Geo_spatial_Encoder"]["dim_in"],
    config["Geo_spatial_Encoder"]["dim_hidden"],
    config["Geo_spatial_Encoder"]["dim_out"],
    config["Geo_spatial_Encoder"]["num_layers"],
)
from Dataloader.Modis_Data_loader.torch_data_loader import Airqualitydataset
from Dataloader.Modis_Data_loader.final_loader import Final_Air_Quality_Dataset_pipeline
from src.Models.neural_process import NeuralProcess


""" In the belowmethod it is hard to make these many types of combinations so we will make a loop of input, output types and flags to create the final datasets."""


"""
  input_vars:
    1: ['latitude', 'longitude','AOD', 'PM2.5']
    2: ['latitude', 'longitude','AOD']
    3: ['latitude', 'longitude','PM2.5']
    4: ['latitude', 'longitude']

# feature_sets

  output_vars:
    1: ['AOD', 'PM2.5']
    2: ['AOD']
    3: ['PM2.5']

"""


input_type = [1, 2, 3, 4]
output_type = [1, 2, 3]
flag = ["Train", "Val", "Test"]

# Now we will loop over all the combinations to create the final datasets.
datasets = {}
for i in input_type:
    if i == "":
        print("Invalid input type")
        continue
    for j in output_type:
        if j == "":
            print("Invalid output type")
            continue
        for k in flag:
            if k == "":
                print("Invalid flag")
                continue
            else:
                # From here we will do the proper string formatting.
                # Here these are the column names below has been mentioned.
                input_name = config["experiments"]["input_vars"][i]
                output_name = config["experiments"]["output_vars"][j]
                print(
                    f"Creating the dataset for Input:{input_name}({i}) | Output: {output_name} ({j}) | Flag: {k}"
                )
                Final_Data = Final_Air_Quality_Dataset_pipeline(
                    config=config,
                    geospatial_encoder=geospatial_encoder,
                    flag=k,
                    input_type=i,
                    output_type=j,
                )
                current_inputs, current_outputs = Final_Data.full_pipeline()

                datasets[(i, j, k)] = {
                    "inputs": current_inputs,
                    "outputs": current_outputs,
                }

# Checking the datsets sizes.
datasets[(1, 2, "Train")]["inputs"].shape
datasets[(1, 2, "Train")]["outputs"].shape
datasets[(1, 2, "Val")]["outputs"].shape
datasets[(1, 1, "Val")]["outputs"].shape
datasets[(1, 2, "Test")]["outputs"].shape
datasets[(1, 1, "Test")]["outputs"].shape

# After preparing the datasets we will train the different models.
# In our case onc ewe are training the model we have inout keys and putput keys are different but the flag is same.
from loss_functions import LossFunctions

Loss = LossFunctions()
from optimizer_utils import NPTrainer
from optimizer_utils import neural_process_data
from optimizer_utils import validation_function
# importing the packages for the optimization.


for input_key in input_type:
    if input_key == "":
        print("Please check the input keys")
        continue
    for output_key in output_type:
        if output_key == "":
            print("Please check the output keys")
            continue
        print(f" Start Training of Input: {input_key} | Output : {output_key}")
        # the current inputs and current outputs are:
        current_inputs = config["experiments"]["input_vars"][input_key]
        in_name = "_".join(current_inputs)
        current_outputs = config["experiments"]["output_vars"][output_key]
        out_name = "_".join(current_outputs)
        # We can calculate the dimensions of the inputs and putputs to decide the number of model's layers.

        x_dim = len(current_inputs)
        y_dim = len(current_outputs)
        # In this model the inputs and outputs are.
        model = NeuralProcess(x_dim, y_dim, x_dim, y_dim, 128, 128).to("cpu")
        # Defining the current check points and log directories.
        current_log_dir = f"./Master-Thesis/cpu_logs/{in_name}_{out_name}"
        current_checkpoint_dir = f"./Master-Thesis/cpu_checkpoints/{in_name}_{out_name}"
        # defining the optimizer
        Optimizer = optim.Adam(
            model.parameters, lr=float(config["Experiments"]["train"]["lr"])
        )
        # Defining the training class
        Training = NPTrainer(
            model=model,
            optimizer=Optimizer,
            loss_fn=Loss,
            device="cpu",
            log_dir=current_log_dir,
            checkpoint_dir=current_checkpoint_dir,
        )
        # To check the model's accuracy in validation data set we will use this.
        best_val_loss = float("inf")
        # these are the training data sets

        NeuralProcess_Train_data = (
            neural_process_data(datasets[(input_key, output_key, "Train")]["inputs"]),
            datasets[(input_key, output_key, "Train")]["outputs"],
        )
        NeuralProcess_Train_dataloader = DataLoader(
            NeuralProcess_Train_data,
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )
        # Now we will define the validation datasets as well as validation dataloasers
        # NeuralProcess_Val_datasets = {}
        NeuralProcess_Val_data = (
            neural_process_data(datasets[(input_key, output_key, "Val")]["inputs"]),
            datasets[(input_key, output_key, "Val")]["outputs"],
        )

        # NeuralProcess_Val_dataloaders = {}
        NeuralProcess_Val_dataloader = DataLoader(
            NeuralProcess_Val_data,
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )

        for epoch in range(100):
            # Train
            train_loss = Training.train_epoch(NeuralProcess_Train_dataloader, epoch)
            val_loss = validation_function(
                model=model,
                val_dataloader=NeuralProcess_Val_dataloader,
                loss=Loss,
                device="cpu",
            )
            print(
                f"Epoch:{epoch} | Val_loss:{val_loss:.4f} | Train_loss:{train_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Training.save_checkpoint(epoch)
                print(f" >>> Saved New Best Model found! ")

# NeuraProcessData_latlon_PM25 = neural_process_data(
#     fnal_data_latlong_PM25[0],
#     fnal_data_latlong_PM25[1],
#     nm_points_per_task=200,
# )


#     dataloader_latlon_PM25 = DataLoader(
#     dataset=NeuralProcessData_latlon_PM25,
#     batch_size=16,
#     shuffle=True,
#     num_workers=0,
# )


# ##################################################################
# # -------------Training of Lat_Lon_PM25 data
# ##################################################################
# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# unique_log_dir = f"./logs/latlon_PM25/{current_time}"
# unique_checkpoint_dir = f"./checkpoints/latlon_PM25/{current_time}"
# Training = NPTrainer(
#     model=model_latlon_PM25,
#     optimizer=Optimizer_latlong_PM25,
#     loss_fn=Loss,
#     device="cpu",
#     log_dir=unique_log_dir,
#     checkpoint_dir=unique_checkpoint_dir,
# )
# #  running the epochs.
# #  ... definitions above ...

# #  Define a "best loss" to track improvement
# best_val_loss = float("inf")
# #
# for epoch in range(100):  # Run for 100 epochs
#     # 1. TRAIN
#     train_loss = Training.train_epoch(dataloader_latlon_PM25, epoch)
#     # 2. VALIDATE (Call the function we just wrote)
#     val_loss = validation_function(
#         model=model_latlon_PM25,  # Access model from your Trainer class
#         val_dataloader=dataloader_val_latlon_PM25,  # You need a separate loader for val data
#         loss_fn=Loss,
#         device="cpu",
#     )
#     print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#     # 3. SAVE ONLY IF BETTER
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         Training.save_checkpoint(epoch)
#         print(f"   >>> SAVED: New best model found!")


from src.Models.neural_process import NeuralProcess

"""
Final models for training........... making attributes from the class
"""

model_latlon_AOD_PM25 = NeuralProcess(128, 2, 128, 2, 128, 128)
model_latlon_AOD = NeuralProcess(127, 2, 127, 2, 128, 128)
model_latlon_PM25 = NeuralProcess(127, 2, 127, 2, 128, 128)
model_latlon = NeuralProcess(126, 2, 126, 2, 128, 128)

Optimizer = optim.Adam(
    model_latlon_AOD_PM25.parameters(), lr=float(config["train"]["lr"])
)
Optimizer_latlong_AOD = optim.Adam(
    model_latlon_AOD.parameters(), lr=float(config["train"]["lr"])
)
Optimizer_latlong_PM25 = optim.Adam(
    model_latlon_PM25.parameters(), lr=float(config["train"]["lr"])
)
Optimizer_latlong = optim.Adam(
    model_latlon.parameters(), lr=float(config["train"]["lr"])
)


import torch.optim as optim
from optimizer_utils import NPTrainer
from loss_functions import LossFunctions

Loss = LossFunctions()

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# we store the logs to store the losses to plot for the tensorboard.
unique_log_dir = f"./logs/latlon/{current_time}"
unique_checkpoint_dir = f"./checkpoints/latlon/{current_time}"

dataloader = DataLoader(
    dataset=NeuralProcessData_latlon_AOD_PM25,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
dataloader.dataset

dataloader_latlon_AOD = DataLoader(
    dataset=NeuralProcessData_latlon_AOD,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

dataloader_latlon_PM25 = DataLoader(
    dataset=NeuralProcessData_latlon_PM25,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
NeuralProcessData_latlon_PM25
dataloader_latlon = DataLoader(
    dataset=NeuralProcessData_latlon,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)
""" Training also we will do in terms of looping to not to get confuse later due to 12 models training."""


# H


Training = NPTrainer(
    model=model_latlon,
    optimizer=Optimizer_latlong,
    loss_fn=Loss,
    device="cpu",
    log_dir=unique_log_dir,
    checkpoint_dir=unique_checkpoint_dir,
)
#  running the epochs.
#  ... definitions above ...

#  Define a "best loss" to track improvement
best_val_loss = float("inf")
#
for epoch in range(10):  # Run for 100 epochs
    # 1. TRAIN
    train_loss = Training.train_epoch(dataloader_latlon, epoch)
    # 2. VALIDATE (Call the function we just wrote)
    val_loss = validation_function(
        model=Training.model,  # Access model from your Trainer class
        val_dataloader=dataloader_val_latlon,  # You need a separate loader for val data
        loss_fn=Loss,
        device="cpu",
    )
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    # 3. SAVE ONLY IF BETTER
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Training.save_checkpoint(epoch)
        print(f"   >>> SAVED: New best model found!")


"""   For type 1 target_columns = ['AOD', 'PM2.5'] & Training dataset      """


# for type 1 target_columns = ['AOD', 'PM2.5'] & Training dataset

#                       Training data

Final_Data_Train_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config,
    geospatial_encoder=geospatial_encoder,
    flag="Train",
    input_type=1,
    output_type=1,
)


Final_Data_Train_latlong_AOD_PM25 = Final_Data_Train_latlong_AOD_PM25.full_pipeline()
Final_Data_Train_latlong_AOD_PM25[0].shape

Final_Data_Train_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=2, output_type=1
)
Final_Data_Train_latlong_AOD = Final_Data_Train_latlong_AOD.full_pipeline()
Final_Data_Train_latlong_AOD[0].shape
Final_Data_Train_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=3, output_type=1
)
Final_Data_Train_latlong_PM25 = Final_Data_Train_latlong_PM25.full_pipeline()
Final_Data_Train_latlong_PM25[0].shape

Final_Data_Train_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=4, output_type=1
)
Final_Data_Train_latlong = Final_Data_Train_latlong.full_pipeline()
Final_Data_Train_latlong[0].shape


#                     Val data


Final_Data_Train_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=1, output_type=1
)
Final_Data_Train_latlong_AOD_PM25 = Final_Data_Train_latlong_AOD_PM25.full_pipeline()
Final_Data_Train_latlong_AOD_PM25[0].shape

Final_Data_Train_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=2, output_type=1
)
Final_Data_Train_latlong_AOD = Final_Data_Train_latlong_AOD.full_pipeline()
Final_Data_Train_latlong_AOD[0].shape
Final_Data_Train_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=3, output_type=1
)
Final_Data_Train_latlong_PM25 = Final_Data_Train_latlong_PM25.full_pipeline()
Final_Data_Train_latlong_PM25[0].shape

Final_Data_Train_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=4, output_type=1
)
Final_Data_Train_latlong = Final_Data_Train_latlong.full_pipeline()
Final_Data_Train_latlong[0].shape


#                          Test data


Final_Data_Train_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=1, output_type=1
)
Final_Data_Train_latlong_AOD_PM25 = Final_Data_Train_latlong_AOD_PM25.full_pipeline()
Final_Data_Train_latlong_AOD_PM25[0].shape

Final_Data_Train_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=2, output_type=1
)
Final_Data_Train_latlong_AOD = Final_Data_Train_latlong_AOD.full_pipeline()
Final_Data_Train_latlong_AOD[0].shape
Final_Data_Train_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=3, output_type=1
)
Final_Data_Train_latlong_PM25 = Final_Data_Train_latlong_PM25.full_pipeline()
Final_Data_Train_latlong_PM25[0].shape

Final_Data_Train_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=4, output_type=1
)
Final_Data_Train_latlong = Final_Data_Train_latlong.full_pipeline()
Final_Data_Train_latlong[0].shape


"""   For type 2 target_columns = ['AOD'] & Training dataset      """


#                       Training data

Final_Data_Train_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=1, output_type=2
)
Final_Data_Train_latlong_AOD_PM25 = Final_Data_Train_latlong_AOD_PM25.full_pipeline()
Final_Data_Train_latlong_AOD_PM25[0].shape

Final_Data_Train_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=2, output_type=2
)
Final_Data_Train_latlong_AOD = Final_Data_Train_latlong_AOD.full_pipeline()
Final_Data_Train_latlong_AOD[0].shape
Final_Data_Train_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=3, output_type=2
)
Final_Data_Train_latlong_PM25 = Final_Data_Train_latlong_PM25.full_pipeline()
Final_Data_Train_latlong_PM25[0].shape

Final_Data_Train_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=4, output_type=2
)
Final_Data_Train_latlong = Final_Data_Train_latlong.full_pipeline()
Final_Data_Train_latlong[0].shape


#                     Val data


Final_Data_Val_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=1, output_type=2
)
Final_Data_Val_latlong_AOD_PM25 = Final_Data_Val_latlong_AOD_PM25.full_pipeline()
Final_Data_Val_latlong_AOD_PM25[0].shape

Final_Data_Val_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=2, output_type=2
)
Final_Data_Val_latlong_AOD = Final_Data_Val_latlong_AOD.full_pipeline()
Final_Data_Val_latlong_AOD[0].shape
Final_Data_Val_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=3, output_type=2
)
Final_Data_Val_latlong_PM25 = Final_Data_Val_latlong_PM25.full_pipeline()
Final_Data_Val_latlong_PM25[0].shape

Final_Data_Val_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=4, output_type=2
)
Final_Data_Val_latlong = Final_Data_Val_latlong.full_pipeline()
Final_Data_Val_latlong[0].shape


#                          Test data


Final_Data_Test_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=1, output_type=2
)
Final_Data_Test_latlong_AOD_PM25 = Final_Data_Test_latlong_AOD_PM25.full_pipeline()
Final_Data_Test_latlong_AOD_PM25[0].shape

Final_Data_Test_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=2, output_type=2
)
Final_Data_Test_latlong_AOD = Final_Data_Test_latlong_AOD.full_pipeline()
Final_Data_Test_latlong_AOD[0].shape
Final_Data_Test_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=3, output_type=2
)
Final_Data_Test_latlong_PM25 = Final_Data_Test_latlong_PM25.full_pipeline()
Final_Data_Test_latlong_PM25[0].shape
Final_Data_Test_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=4, output_type=2
)
Final_Data_Test_latlong = Final_Data_Test_latlong.full_pipeline()
Final_Data_Test_latlong[0].shape


"""   For type 3 target_columns = ['PM2.5'] & Training dataset      """

#                       Training data

Final_Data_Train_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=1, output_type=3
)
Final_Data_Train_latlong_AOD_PM25 = Final_Data_Train_latlong_AOD_PM25.full_pipeline()
Final_Data_Train_latlong_AOD_PM25[0].shape

Final_Data_Train_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=2, output_type=3
)
Final_Data_Train_latlong_AOD = Final_Data_Train_latlong_AOD.full_pipeline()
Final_Data_Train_latlong_AOD[0].shape
Final_Data_Train_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=3, output_type=3
)
Final_Data_Train_latlong_PM25 = Final_Data_Train_latlong_PM25.full_pipeline()
Final_Data_Train_latlong_PM25[0].shape

Final_Data_Train_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Train", input_type=4, output_type=3
)
Final_Data_Train_latlong = Final_Data_Train_latlong.full_pipeline()
Final_Data_Train_latlong[0].shape


#                     Val data


Final_Data_Val_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=1, output_type=3
)
Final_Data_Val_latlong_AOD_PM25 = Final_Data_Train_latlong_AOD_PM25.full_pipeline()
Final_Data_Val_latlong_AOD_PM25[0].shape

Final_Data_Val_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=2, output_type=3
)
Final_Data_Val_latlong_AOD = Final_Data_Val_latlong_AOD.full_pipeline()
Final_Data_Val_latlong_AOD[0].shape
Final_Data_Val_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=3, output_type=3
)
Final_Data_Val_latlong_PM25 = Final_Data_Val_latlong_PM25.full_pipeline()
Final_Data_Val_latlong_PM25[0].shape
Final_Data_Val_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Val", input_type=4, output_type=3
)
Final_Data_Val_latlong = Final_Data_Val_latlong.full_pipeline()
Final_Data_Val_latlong[0].shape


#                          Test data


Final_Data_Test_latlong_AOD_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=1, output_type=3
)
Final_Data_Test_latlong_AOD_PM25 = Final_Data_Test_latlong_AOD_PM25.full_pipeline()
Final_Data_Test_latlong_AOD_PM25[0].shape

Final_Data_Test_latlong_AOD = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=2, output_type=3
)
Final_Data_Test_latlong_AOD = Final_Data_Test_latlong_AOD.full_pipeline()
Final_Data_Test_latlong_AOD[0].shape
Final_Data_Test_latlong_PM25 = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=3, output_type=3
)
Final_Data_Test_latlong_PM25 = Final_Data_Test_latlong_PM25.full_pipeline()
Final_Data_Test_latlong_PM25[0].shape
Final_Data_Test_latlong = Final_Air_Quality_Dataset_pipeline(
    config=config, flag="Test", input_type=4, output_type=3
)
Final_Data_Test_latlong = Final_Data_Test_latlong.full_pipeline()
Final_Data_Test_latlong[0].shape


# we will lets down the final data sets


#                       Validation data


# .........................................................

from Dataloader.Modis_Data_loader.torch_data_loader import (
    AirQualityDataset_latlon_AOD_PM25,
    AirQualityDataset_latlon_AOD,
    AirQualityDataset_latlon_PM25,
    AirQualityDataset_latlon,
    Airqualitydataset,
)


# importing config files
from Dataloader.Modis_Data_loader.final_loader import (
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25,
    Final_Air_Quality_Dataset_pipeline_latlon_AOD,
    Final_Air_Quality_Dataset_pipeline_latlon_PM25,
    Final_Air_Quality_Dataset_pipeline_latlon,
    Final_Air_Quality_Dataset_pipeline,
)

"""
Validation data set 
"""


from Dataloader.Modis_Data_loader.final_loader import (
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25_val,
    Final_Air_Quality_Dataset_pipeline_latlon_AOD_val,
    Final_Air_Quality_Dataset_pipeline_latlon_PM25_val,
    Final_Air_Quality_Dataset_pipeline_latlon_val,
)


"""
Test data set 
"""
from Dataloader.Modis_Data_loader.final_loader import (
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
final_data_latlong_AOD_val = instance_latlon_AOD_val.full_pipeline()
final_data_latlong_AOD_val[0].shape


# latlon_PM2.5
instance_latlon_PM25_val.modis_data_sets()
instance_latlon_PM25_val.stations_data_sets()
instance_latlon_PM25_val.PM_25_data()
instance_latlon_PM25_val.training_data()
instance_latlon_PM25_val.Torch_data()
final_data_latlong_PM25_val = instance_latlon_PM25_val.full_pipeline()
final_data_latlong_PM25_val[0].shape


# latlon
instance_latlon_val.modis_data_sets()
instance_latlon_val.stations_data_sets()
instance_latlon_val.PM_25_data()
instance_latlon_val.training_data()
instance_latlon_val.Torch_data()
final_data_latlong_val = instance_latlon_val.full_pipeline()
final_data_latlong_val[0].shape

# ************************ Final validation data ***************************************
final_data_latlong_AOD_PM25_val[0].shape
final_data_latlong_AOD_val[0].shape
final_data_latlong_PM25_val[0].shape
final_data_latlong_val[0].shape
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
model_latlon = NeuralProcess(126, 2, 126, 2, 128, 128)


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

float(float(config["train"]["lr"]))
# here we are using ADAM optimizer.

Optimizer = optim.Adam(
    model_latlon_AOD_PM25.parameters(), lr=float(config["train"]["lr"])
)
Optimizer_latlong_AOD = optim.Adam(
    model_latlon_AOD.parameters(), lr=float(config["train"]["lr"])
)
Optimizer_latlong_PM25 = optim.Adam(
    model_latlon_PM25.parameters(), lr=float(config["train"]["lr"])
)
Optimizer_latlong = optim.Adam(
    model_latlon.parameters(), lr=float(config["train"]["lr"])
)

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


NeuralProcessData_latlon_AOD = neural_process_data(
    final_data_latlong_AOD[0],
    final_data_latlong_AOD[1],
    num_points_per_task=200,
)

# for test data

NeuralProcessData_latlon_AOD_val = neural_process_data(
    final_data_latlong_AOD_val[0],
    final_data_latlong_AOD_val[1],
    num_points_per_task=200,
)

NeuralProcessData_latlon_PM25 = neural_process_data(
    final_data_latlong_PM25[0],
    final_data_latlong_PM25[1],
    num_points_per_task=200,
)

# for test data

NeuralProcessData_latlon_PM25_val = neural_process_data(
    final_data_latlong_PM25_val[0],
    final_data_latlong_PM25_val[1],
    num_points_per_task=200,
)

NeuralProcessData_latlon = neural_process_data(
    final_data_latlong[0],
    final_data_latlong[1],
    num_points_per_task=200,
)

# for test data

NeuralProcessData_latlon_val = neural_process_data(
    final_data_latlong_val[0],
    final_data_latlong_val[1],
    num_points_per_task=200,
)


# ------------------------------- practice --------------------

X_task, Y_task = NeuralProcessData_latlon_AOD[0]
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

dataloader_latlon_AOD = DataLoader(
    dataset=NeuralProcessData_latlon_AOD,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

dataloader_latlon_PM25 = DataLoader(
    dataset=NeuralProcessData_latlon_PM25,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

dataloader_latlon = DataLoader(
    dataset=NeuralProcessData_latlon,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

for x, y in dataloader_latlon_PM25:
    x.shape, y.shape

for x, y in dataloader_latlon_AOD:
    x.shape, y.shape

for x, y in dataloader_latlon:
    x.shape, y.shape


# checking the size and shape of data loader
for x, y in dataloader:
    print(x.shape, y.shape)
len(dataloader)


# ---------------------  Creating the validation data loaders --------------------------------


# creating data loader fo rvalidation data
dataloader_val = DataLoader(
    dataset=NeuralProcessData_latlon_AOD_PM25_val,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)


dataloader_val_latlon_AOD = DataLoader(
    dataset=NeuralProcessData_latlon_AOD_val,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)


dataloader_val_latlon_PM25 = DataLoader(
    dataset=NeuralProcessData_latlon_PM25_val,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

dataloader_val_latlon = DataLoader(
    dataset=NeuralProcessData_latlon_val,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)


# checking the size and shape of data loader
for x, y in dataloader_val:
    print(x.shape, y.shape)
len(dataloader_val)


for x, y in dataloader_val_latlon_AOD:
    x.shape, y.shape


for x, y in dataloader_val_latlon_PM25:
    x.shape, y.shape

for x, y in dataloader_val_latlon:
    x.shape, y.shape
# In this way we can extract the x, y batches from dataloader.

"""
How to check the training dataloader 
"""

len(dataloader)
Xb, Yb = next(iter(dataloader))
Xb.shape, Yb.shape


"""
Start training for all the models in their respective data sets 
"""

# importing the trainer
from optimizer_utils import NPTrainer
from optimizer_utils import validation_function

# cretaing and saving the logs as per the current training.
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
unique_log_dir = f"./logs/latlon_AOD_PM25/{current_time}"
unique_checkpoint_dir = f"/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Master-Thesis/checkpoints/latlon_AOD/{current_time}"

## 1 -----------------------  Lat_Lon_AOD_PM2.5 data -----------------------------
##################################################################################
# Training the model
#
Training = NPTrainer(
    model=model_latlon_AOD_PM25,
    optimizer=Optimizer,
    loss_fn=Loss,
    device="cpu",
    log_dir=unique_log_dir,
    checkpoint_dir=unique_checkpoint_dir,
)

# running the epochs.
# ... definitions above ...
#
# Define a "best loss" to track improvement
best_val_loss = float("inf")

for epoch in range(10):  # Run for 100 epochs
    # 1. TRAIN
    train_loss = Training.train_epoch(dataloader, epoch)

    # 2. VALIDATE (Call the function we just wrote)
    val_loss = validation_function(
        model=Training.model,  # Access model from your Trainer class
        val_dataloader=dataloader_val,  # You need a separate loader for val data
        loss_fn=Loss,
        device="cpu",
    )

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 3. SAVE ONLY IF BETTER
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Training.save_checkpoint(epoch)
        print(f"   >>> SAVED: New best model found!")


##################################################################################
# """
# Now we need to export the validation data-set so that our trained models can be validated and model with best weights can be selected.
# """


# self, model, val_dataloader, device, Loss, context_target_split
# Now we are extracting the weights and will be using it for the testing purpose we will make our model ready for test.


##################################################################
# ----------- Training of Lat_Lon_AOD data ------------
##################################################################

# unique_log_dir = f"./logs/latlon_AOD/{current_time}"
# unique_checkpoint_dir = f"./checkpoints/latlon_AOD/{current_time}"
# Training = NPTrainer(
#     model=model_latlon_AOD,
#     optimizer=Optimizer_latlong_AOD,
#     loss_fn=Loss,
#     device="cpu",
#     log_dir=unique_log_dir,
#     checkpoint_dir=unique_checkpoint_dir,
# )
# #  running the epochs.
# #  ... definitions above ...

# #  Define a "best loss" to track improvement
# best_val_loss = float("inf")
# #
# for epoch in range(100):  # Run for 100 epochs
#     # 1. TRAIN
#     train_loss = Training.train_epoch(dataloader_latlon_AOD, epoch)
#     # 2. VALIDATE (Call the function we just wrote)
#     val_loss = validation_function(
#         model=model_latlon_AOD,  # Access model from your Trainer class
#         val_dataloader=dataloader_val_latlon_AOD,  # You need a separate loader for val data
#         loss_fn=Loss,
#         device="cpu",
#     )
#     print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#     # 3. SAVE ONLY IF BETTER
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         Training.save_checkpoint(epoch)
#         print(f"   >>> SAVED: New best model found!")


# ##################################################################
# # -------------Training of Lat_Lon_PM25 data
# ##################################################################
# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# unique_log_dir = f"./logs/latlon_PM25/{current_time}"
# unique_checkpoint_dir = f"./checkpoints/latlon_PM25/{current_time}"
# Training = NPTrainer(
#     model=model_latlon_PM25,
#     optimizer=Optimizer_latlong_PM25,
#     loss_fn=Loss,
#     device="cpu",
#     log_dir=unique_log_dir,
#     checkpoint_dir=unique_checkpoint_dir,
# )
# #  running the epochs.
# #  ... definitions above ...

# #  Define a "best loss" to track improvement
# best_val_loss = float("inf")
# #
# for epoch in range(100):  # Run for 100 epochs
#     # 1. TRAIN
#     train_loss = Training.train_epoch(dataloader_latlon_PM25, epoch)
#     # 2. VALIDATE (Call the function we just wrote)
#     val_loss = validation_function(
#         model=model_latlon_PM25,  # Access model from your Trainer class
#         val_dataloader=dataloader_val_latlon_PM25,  # You need a separate loader for val data
#         loss_fn=Loss,
#         device="cpu",
#     )
#     print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#     # 3. SAVE ONLY IF BETTER
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         Training.save_checkpoint(epoch)
#         print(f"   >>> SAVED: New best model found!")


################################################################
# Training on lat_lon data set
##################################################################
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# we store the logs to store the losses to plot for the tensorboard.
unique_log_dir = f"./logs/latlon/{current_time}"
unique_checkpoint_dir = f"./checkpoints/latlon/{current_time}"
Training = NPTrainer(
    model=model_latlon,
    optimizer=Optimizer_latlong,
    loss_fn=Loss,
    device="cpu",
    log_dir=unique_log_dir,
    checkpoint_dir=unique_checkpoint_dir,
)
#  running the epochs.
#  ... definitions above ...

#  Define a "best loss" to track improvement
best_val_loss = float("inf")
#
for epoch in range(10):  # Run for 100 epochs
    # 1. TRAIN
    train_loss = Training.train_epoch(dataloader_latlon, epoch)
    # 2. VALIDATE (Call the function we just wrote)
    val_loss = validation_function(
        model=Training.model,  # Access model from your Trainer class
        val_dataloader=dataloader_val_latlon,  # You need a separate loader for val data
        loss_fn=Loss,
        device="cpu",
    )
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    # 3. SAVE ONLY IF BETTER
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        Training.save_checkpoint(epoch)
        print(f"   >>> SAVED: New best model found!")


# Loadign the model and then plottign the curvs..............
