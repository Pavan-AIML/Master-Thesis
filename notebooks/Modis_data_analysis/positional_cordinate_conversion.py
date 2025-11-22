# with the help of this file we can import the modules from Thesis folder

"""
Importing necessary libraries
"""

import os
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PM25_data_loader_analysis import Modis_data_loader
from PM25_data_loader_analysis import PM_25_dataloader
from PM25_data_loader_analysis import combine_the_data_frames
from PM25_data_loader_analysis import Training_data_loader

# sys.path.append(str(Path(__file__).resolve().parents[2] / "locationencoder"))
# s
# ys.path.append("../../Master-Thesis/")
from torch_data_loader import AirQualityDataset
from utils import config
import arrow
from datetime import datetime

# -------------************---------------------------------


"""
Making the Master thesis folder as root folder for imports 
"""


ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)


# -------------************---------------------------------


# -------------************---------------------------------


"""
Loading PM2.5 and AOD data set finally in torch data loader format

"""


Modis_data_2018 = Modis_data_loader("MODIS_AOD/merged_data_2018_sorted_dates.csv")
Modis_data_2018.get_head()


Modis_data_2019 = Modis_data_loader("MODIS_AOD/merged_data_2019_sorted_dates.csv")
Modis_data_2019.get_head()


Modis_data_2020 = Modis_data_loader("MODIS_AOD/merged_data_2020_sorted_dates.csv")
Modis_data_2020.get_head()


Modis_data_2021 = Modis_data_loader("MODIS_AOD/merged_data_2021_sorted_dates.csv")
Modis_data_2021.get_head()

Modis_data_2022 = Modis_data_loader("MODIS_AOD/merged_data_2022_sorted_dates.csv")
Modis_data_2022.get_head()


Modis_data_2023 = Modis_data_loader("MODIS_AOD/merged_data_2023_sorted_dates.csv")
Modis_data_2023.get_head()

Lat_Long_data = Modis_data_loader("Stations_Lat_Long/stn_extrafeat.csv")

Lat_Long_data.data

stations = Lat_Long_data.data[["Latitude", "Longitude"]]
stations

PM_data_instance = PM_25_dataloader()
PM_data_instance.get_shape()

PM_data = PM_25_dataloader()
PM_data = PM_data.get_data()

PM_data.shape
# Training data

loader = Training_data_loader(
    Modis_data_2020.data, Modis_data_2021.data, Modis_data_2022.data, PM_data, stations
)
loader.daily_PM25_data_extraction()

loader.AOD_data_fusion()
loader.PM25_nearest_neighbour_finder(loader.AOD_data_fusion())

loader.data_cleaning_and_missing_values_handeling(
    loader.PM25_nearest_neighbour_finder(loader.AOD_data_fusion())
)


df = loader.get_final_training_data(
    loader.data_cleaning_and_missing_values_handeling(
        loader.PM25_nearest_neighbour_finder(loader.AOD_data_fusion())
    ),
    loader.daily_PM25_data_extraction(),
)
df.columns

final_training_data = loader.prepare_final_trainig_data()
final_training_data.shape

# torch data loader instance

Instance = AirQualityDataset(final_training_data, config)


final_training_data = AirQualityDataset(final_training_data, config)


lonlat = final_training_data.__getitem__(10)[0][0:2]
lonlat = lonlat.unsqueeze(0)

# loading the torch data set

# -------------************---------------------------------

"""
our final training data is the data in which we will train our neural network model.

"""

from locationencoder.final_location_encoder import Geospatial_Encoder


"""
Making instance of the Geospatial_Encoder
"""

# as the legendre polyomials of degree 10 hen the output of spherical harmonics will be 121
# hence the input dimension to siren will be 121
# hence we will take dim_in as 121
# outpur dim is 4
geospatial_encoder_instance = Geospatial_Encoder(
    config["Geo_spatial_Encoder"]["dim_in"],
    config["Geo_spatial_Encoder"]["dim_hidden"],
    config["Geo_spatial_Encoder"]["dim_out"],
    config["Geo_spatial_Encoder"]["num_layers"],
)


lonlat_embedding = geospatial_encoder_instance(lonlat)

lonlat_embedding
