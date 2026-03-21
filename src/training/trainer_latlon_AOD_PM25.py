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
# from loss_functions import NPELBO
from datetime import datetime
global_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
# from utils.config import config
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)
from src.training.optimizer_utils import NPTrainer, neural_process_data
from src.training.loss_functions import NPELBO

device = torch.device('cuda')
device
def seed_everything(seed = 42):
    import random
    import os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()
from locationencoder.final_location_encoder import Geospatial_Encoder

from configs.utils import config

current_outputs = config["experiments"]["output_vars"][1]
current_outputs
import pytorch_lightning as pl

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



# Defining the geospatial encoder outside of the argument as it should not coming again and again.


geospatial_encoder = Geospatial_Encoder(
    config["Geo_spatial_Encoder"]["dim_in"],
    config["Geo_spatial_Encoder"]["dim_hidden"],
    config["Geo_spatial_Encoder"]["dim_out"],
    config["Geo_spatial_Encoder"]["num_layers"],
).to("cpu")

from Dataloader.Modis_Data_loader.torch_data_loader import Airqualitydataset
from Dataloader.Modis_Data_loader.final_loader import Final_Air_Quality_Dataset_pipeline
from src.Models.neural_process import NeuralProcess



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
def seed_everything(seed = 42):
    import random
    import os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()




""" Data set loop creating """
# Now we will loop over all the combinations to create the final datasets.
# """################################################"""
# "check the pipeline where is the problem "

# Final_Data = Final_Air_Quality_Dataset_pipeline(
#                     config=config,
#                     geospatial_encoder=geospatial_encoder.to(device),
#                     flag="Train",
#                     input_type=1,
#                     output_type=1,
#                 )

# Final_Data("MODIS_AOD/merged_data_2018_sorted_dates.csv")
input_type = [8, 10, 11, 12]  # 5 = with meteorological features (rh, ws, wd, temp, press)
output_type = [1, 2, 3]
flag = ["Train", "Val", "Test"]


""""############################"""
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
                    geospatial_encoder=geospatial_encoder.to(device),
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

# datasets[(input_types, output_type, flag)]

# datasets[(1, 2, "Train")]["inputs"].shape
# datasets[(1, 2, "Val")]["inputs"].shape
# datasets[(1, 2, "Test")]["inputs"].shape
# datasets[(1, 2, "Train")]["outputs"].shape
# datasets[(1, 2, "Val")]["outputs"].shape
# datasets[(1, 1, "Val")]["outputs"].shape
# datasets[(1, 2, "Test")]["outputs"].shape
# datasets[(1, 1, "Test")]["outputs"].shape
# datasets[(1, 1, "Train")]["inputs"].shape
# datasets[(1, 1, "Train")]["outputs"].shape

# # After preparing the datasets we will train the different models.
# # In our case onc ewe are training the model we have inout keys and putput keys are different but the flag is same.
# datasets[(1, 2, "Val")]["outputs"]
# len(config["experiments"]["input_vars"][1])
# len(config["experiments"]["output_vars"][1])

# float(config["train"]["lr"])
# len(datasets[(1, 1, "Train")]["inputs"][0])
# len(datasets[(1, 1, "Train")]["outputs"][0])
# datasets[(1, 1, "Train")]["outputs"]



"Here is the new one we are using "

for input_key in input_type:
    if input_key == "":
        print("Please check the input keys")
        continue
    for output_key in output_type:
        if output_key == "":
            print("Please check the output keys")
            continue
        print(f" Start Training of Input: {input_key} | Output : {output_key}")
        # Input / output column names from config
        current_inputs = config["experiments"]["input_vars"][input_key]
        in_name = "_".join(current_inputs)
        current_outputs = config["experiments"]["output_vars"][output_key]
        out_name = "_".join(current_outputs)
        foldername = f"{in_name}____{out_name}"
        # Extract training tensors
        train_inputs = datasets[(input_key, output_key, "Train")]["inputs"]   # [N_train, x_dim]
        train_outputs = datasets[(input_key, output_key, "Train")]["outputs"] # [N_train, y_dim]
        print(f"DEBUG: Total rows for {foldername}: {len(train_outputs)}")
        # Loss (DeepMind-style ELBO)
        Loss = NPELBO(beta=1.0)
        # Compute target mean/std for de-normalization in RMSE/R2
        # train_outputs should be a tensor [N, y_dim] or [N_tasks, N_points, y_dim]
        if train_outputs.dim() == 2:
            # [N, y_dim]
            t_mean = train_outputs.mean(dim=0)
            t_std = train_outputs.std(dim=0)
        else:
            # e.g. [N_tasks, N_points, y_dim]
            t_mean = train_outputs.mean(dim=(0, 1))
            t_std = train_outputs.std(dim=(0, 1))
        x_dim = train_inputs.shape[-1]
        y_dim = train_outputs.shape[-1]
        print(f"input dim : {x_dim}, output_dim : {y_dim}")
        print(f"Input columns : {current_inputs}, output columns: {current_outputs}")
        # Neural Process model: (x_dim, y_dim, hidden_dim, latent_dim)
        model = NeuralProcess(x_dim, y_dim, hidden_dim=128, latent_dim=128).to(device)
        # current_time = datetime.now().strftime("%Y-%m-%d__%H:%M")
        # Log / checkpoint directories
        current_log_dir = ROOT / "logs_20km" / foldername / global_run_id
        current_checkpoint_dir = ROOT / "Gpu_checkpoints_tum" / foldername / global_run_id
        # Optimizer
        Optimizer = optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
        # Trainer
        Training = NPTrainer(
            model=model.to(device),
            optimizer=Optimizer,
            loss_fn=Loss,
            device=device,
            log_dir=current_log_dir,
            checkpoint_dir=current_checkpoint_dir,
            input_dim=x_dim,
            output_dim=y_dim,
            context_min=50,
            context_max=100,
            num_target=200,
            target_mean=t_mean,
            target_std=t_std,
        )
        best_val_loss = float("inf")
        # Build NP datasets (each item is a "task" of 25 points)
        NeuralProcess_Train_data = neural_process_data(
            datasets[(input_key, output_key, "Train")]["inputs"],
            datasets[(input_key, output_key, "Train")]["outputs"],
            num_points_per_task=25,
        )
        NeuralProcess_Train_dataloader = DataLoader(
            NeuralProcess_Train_data,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        NeuralProcess_Val_data = neural_process_data(
            datasets[(input_key, output_key, "Val")]["inputs"],
            datasets[(input_key, output_key, "Val")]["outputs"],
            num_points_per_task=25,
        )
        NeuralProcess_Val_dataloader = DataLoader(
            NeuralProcess_Val_data,
            batch_size=16,
            shuffle=False,
            num_workers=4,  # num of workers should be > 0 if pin_memory = True
            pin_memory=True,
            persistent_workers=True,
        )
        for epoch in range(100):
            # Train
            train_loss = Training.train_epoch(
                NeuralProcess_Train_dataloader,
                epoch,
                target_col_names=current_outputs,
            )
            val_loss = Training.evaluate(NeuralProcess_Val_dataloader)
            Training.writer.add_scalar("Loss/Validation", val_loss, epoch)
            print(
                f"Epoch:{epoch} | Val_loss:{val_loss:.4f} | Train_loss:{train_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Training.save_checkpoint(epoch)
                print(" >>> Saved New Best Model found! ")
        Training.log_final_hparams(best_val_loss)
        print(f"Finished Training for {foldername}")
        Training.writer.close()


