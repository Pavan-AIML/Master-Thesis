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
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import arrow
from datetime import datetime


ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)
# -------------************---------------------------------
# loading all the data loaders

from notebooks.Modis_data_analysis.PM25_data_loader_analysis import Modis_data_loader

from notebooks.Modis_data_analysis.PM25_data_loader_analysis import PM_25_dataloader
from notebooks.Modis_data_analysis.PM25_data_loader_analysis import (
    combine_the_data_frames,
)
from notebooks.Modis_data_analysis.PM25_data_loader_analysis import Training_data_loader
from locationencoder.final_location_encoder import Geospatial_Encoder
from notebooks.Modis_data_analysis.torch_data_loader import AirQualityDataset

# importing config files
from notebooks.Modis_data_analysis.utils import config
from torch.utils.data import DataLoader


# -------------************---------------------------------


# -------------************---------------------------------


class Final_Air_Quality_Dataset_pipeline:
    def __init__(self, config):
        self.config = config
        self.Modis_data_2020 = None
        self.Modis_data_2021 = None
        self.Modis_data_2022 = None
        self.Modeis_data_2022 = None
        self.Modis_data_2023 = None
        self.stations = None
        self.PM_data = None
        self.Training_data = None
        self.torch_data = None
        self.geospatial_encoder_instance = Geospatial_Encoder(
            config["Geo_spatial_Encoder"]["dim_in"],
            config["Geo_spatial_Encoder"]["dim_hidden"],
            config["Geo_spatial_Encoder"]["dim_out"],
            config["Geo_spatial_Encoder"]["num_layers"],
        )

    def modis_data_sets(self):
        self.Modis_data_2018 = Modis_data_loader(
            "MODIS_AOD/merged_data_2018_sorted_dates.csv"
        )
        self.Modis_data_2019 = Modis_data_loader(
            "MODIS_AOD/merged_data_2019_sorted_dates.csv"
        )
        self.Modis_data_2020 = Modis_data_loader(
            "MODIS_AOD/merged_data_2020_sorted_dates.csv"
        )
        self.Modis_data_2021 = Modis_data_loader(
            "MODIS_AOD/merged_data_2021_sorted_dates.csv"
        )
        self.Modis_data_2022 = Modis_data_loader(
            "MODIS_AOD/merged_data_2022_sorted_dates.csv"
        )
        self.Modis_data_2023 = Modis_data_loader(
            "MODIS_AOD/merged_data_2023_sorted_dates.csv"
        )

        return [
            self.Modis_data_2020.data,
            self.Modis_data_2021.data,
            self.Modis_data_2022.data,
        ]

    def stations_data_sets(self):
        Lat_Long_data = Modis_data_loader("Stations_Lat_Long/stn_extrafeat.csv")
        self.stations = Lat_Long_data.data[["Latitude", "Longitude"]]
        return self.stations

    def PM_25_data(self):
        PM_data = PM_25_dataloader()
        self.PM_data = PM_data.get_data()
        return self.PM_data

    def training_data(self):
        loader = Training_data_loader(
            self.Modis_data_2020.data,
            self.Modis_data_2021.data,
            self.Modis_data_2022.data,
            self.PM_data,
            self.stations,
        )
        self.Training_data = loader.prepare_final_trainig_data()
        return self.Training_data

    def Torch_data(self):
        self.torch_data = AirQualityDataset(self.Training_data, self.config)
        return self.torch_data

    def full_pipeline(self):
        torch_latlong_loader = DataLoader(
            self.torch_data, batch_size=len(self.torch_data)
        )
        for inputs, outputs in torch_latlong_loader:
            all_latlong = inputs[:, 0:2]
        all_latitude_longitude_embeddings = self.geospatial_encoder_instance(
            all_latlong
        )
        final_inputs = torch.cat(
            (all_latitude_longitude_embeddings, inputs[:, 2:]), dim=1
        )

        return final_inputs, outputs
