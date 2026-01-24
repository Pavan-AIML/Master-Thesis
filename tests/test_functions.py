# with the help of this file we can import the modules from Thesis folder

import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# sys.path.append("../../Master-Thesis/")
"""
# Making the Master thesis folder as root folder for imports 

"""
ROOT = Path(__file__).resolve().parents[1]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)
# importing the Modis data loader module
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import Modis_data_loader

from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import (
    Training_data_loader,
)


# Importing the PM 2.5 data loader module
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import PM_25_dataloader

# All the data sets for testing
Modis_data_1 = Modis_data_loader("MODIS_AOD/merged_data_2018_sorted_dates.csv").data


Modis_data_2 = Modis_data_loader("MODIS_AOD/merged_data_2018_sorted_dates.csv").data

Modis_data_3 = Modis_data_loader("MODIS_AOD/merged_data_2018_sorted_dates.csv").data

PM_25_data = PM_25_dataloader("PM_2.5_Data/3_hr_data_Sept_2023_v1.npy").get_data()

PM_25_data.head(), PM_25_data.shape, type(PM_25_data)

Stn_data = Modis_data_loader("Stations_Lat_Long/stn_extrafeat.csv").data
Stn_data = Stn_data[["Latitude", "Longitude"]]

Stn_data.head(), Stn_data.shape, type(Stn_data)


# Lets extract all the functions in the data loader and test them here individually to tackle any bugs if present.

#
Training_data_instance = Training_data_loader(
    Modis_data_1, Modis_data_2, Modis_data_3, PM_25_data, Stn_data
)
# Yearly PM2.5 data


import inspect

print(
    [
        name
        for name, _ in inspect.getmembers(
            Training_data_instance, predicate=inspect.ismethod
        )
    ]
)

daily_PM25_data = Training_data_instance.daily_PM25_data_extraction()
daily_PM25_data.head(), daily_PM25_data.shape, type(daily_PM25_data)

Complete_AOD_data = Training_data_instance.AOD_data_fusion()
Complete_AOD_data.head(), Complete_AOD_data.shape, type(Complete_AOD_data)

Nearest_neighbours_data = Training_data_instance.PM25_nearest_neighbour_finder(
    Complete_AOD_data
)

(
    Nearest_neighbours_data.head(),
    Nearest_neighbours_data.shape,
    type(Nearest_neighbours_data),
)

Cleaned_AOD_data = Training_data_instance.data_cleaning_and_missing_values_handeling(
    Nearest_neighbours_data
)
Cleaned_AOD_data.head(), Cleaned_AOD_data.shape, type(Cleaned_AOD_data)

Final_training_data = Training_data_instance.get_final_training_data(
    Cleaned_AOD_data, daily_PM25_data
)

Final_training_data.head(), Final_training_data.shape, type(Final_training_data)

import sys
from pathlib import Path
import glob

ROOT = Path(__file__).resolve().parents[1]
print(ROOT)

# here we will check our data_loader individually


def finding_the_best_model_checkpoint(best_experiment_folder_path):
    # First of all we will go and look for the best experiment folder path.
    best_experiment_path = ROOT / "cpu_checkpoints" / best_experiment_folder_path

    if not os.path.exists(best_experiment_path):
        raise FileExistsError(
            f"Best experiment folder path not found:{best_experiment_path}"
        )

    # Now we will go and look for the latest time stamp as we have saved our latest directory in it

    time_stamp_folder = glob.glob(os.path.join(best_experiment_path, "*"))

    if not time_stamp_folder:
        raise FileExistsError(
            f"No time stamp folder found in the {best_experiment_path}"
        )
    latest_time_stamp_folder = max(time_stamp_folder, key=os.path.getmtime)

    # Now once we have found the latest time stamp foder we will look inside and find the best model checkpoint.

    best_check_point_dirs = glob.glob(os.path.join(latest_time_stamp_folder, "*.pth"))
    if not best_check_point_dirs:
        raise FileExistsError(f"No check-point file found {best_check_point_dirs}")

    # Now as we have found the best check-points we will return the check-point with highest number of epoch.
    
    # to find the best check point we will use the lambda function. 
    # Here the logic of the lambda function comes from to check the check-point file in the order of the number written in the check point file name.
    
    best_check_point_file = max(best_check_point_dirs, key = lambda f : int(f.split("_")[-1].split(".")[0]))
    

    return best_check_point_file


finding_the_best_model_checkpoint("latitude_longitude_AOD_PM2.5____AOD_PM2.5")
