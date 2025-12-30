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

from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import Modis_data_loader

from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import PM_25_dataloader

from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import (
    combine_the_data_frames,
)

from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import (
    Training_data_loader,
)
from locationencoder.final_location_encoder import Geospatial_Encoder
from Dataloader.Modis_Data_loader.torch_data_loader import (
    AirQualityDataset_latlon_AOD_PM25,
    AirQualityDataset_latlon_PM25,
    AirQualityDataset_latlon_AOD,
    AirQualityDataset_latlon,
)


# importing config files
from configs.utils import config
from torch.utils.data import DataLoader

from configs import config
# -------------************---------------------------------


class Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25:
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
        self.torch_data = AirQualityDataset_latlon_AOD_PM25(
            self.Training_data, self.config
        )
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


# **************** class for validation data set
class Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25_val:
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
        self.torch_data = AirQualityDataset_latlon_AOD_PM25(
            self.Training_data, self.config, flag="Val"
        )
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


# *************************** Test data pipeline *************************************


class Final_Air_Quality_Dataset_pipeline_latlon_AOD_PM25_test:
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
        self.torch_data = AirQualityDataset_latlon_AOD_PM25(
            self.Training_data, self.config, flag="Test"
        )
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


# ------------------------------------------------------------------------------------
# Lat_lon_AOD
# ------------------------------------------------------------------------------------

# ******************************** Train data pipeline *********************************


class Final_Air_Quality_Dataset_pipeline_latlon_AOD:
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
        self.torch_data = AirQualityDataset_latlon_AOD(self.Training_data, self.config)
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


# ******************************* validation data pipeline *****************************


class Final_Air_Quality_Dataset_pipeline_latlon_AOD_val:
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
        self.torch_data = AirQualityDataset_latlon_AOD(
            self.Training_data, self.config, flag="Val"
        )
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


# ************************************** Test data pipeline ***************************


class Final_Air_Quality_Dataset_pipeline_latlon_AOD_test:
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
        self.torch_data = AirQualityDataset_latlon_AOD(
            self.Training_data, self.config, flag="Test"
        )
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


# ----------------------------------------------------------------------------------------
# Lat_long_PM2.5
# ----------------------------------------------------------------------------------------

# ********************************* Train data pipeline ********************************


class Final_Air_Quality_Dataset_pipeline_latlon_PM25:
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
        self.torch_data = AirQualityDataset_latlon_PM25(self.Training_data, self.config)
        return self.torch_data

    def full_pipeline(self):
        torch_latlong_loader = DataLoader(
            self.torch_data, batch_size=len(self.torch_data)
        )
        for inputs, outputs in torch_latlong_loader:
            all_latlong = inputs[:, 0:2]
        with torch.no_grad():
            all_latitude_longitude_embeddings = self.geospatial_encoder_instance(
                all_latlong
            )
        final_inputs = torch.cat(
            (all_latitude_longitude_embeddings, inputs[:, 2:]), dim=1
        )
        final_inputs = final_inputs.clone().detach()
        outputs = outputs.clone().detach()
        return final_inputs, outputs


# *************************** val **************\\


class Final_Air_Quality_Dataset_pipeline_latlon_PM25_val:
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
        self.torch_data = AirQualityDataset_latlon_PM25(
            self.Training_data, self.config, flag="Val"
        )
        return self.torch_data

    def full_pipeline(self):
        torch_latlong_loader = DataLoader(
            self.torch_data, batch_size=len(self.torch_data)
        )
        for inputs, outputs in torch_latlong_loader:
            all_latlong = inputs[:, 0:2]
        with torch.no_grad():
            all_latitude_longitude_embeddings = self.geospatial_encoder_instance(
                all_latlong
            )
        final_inputs = torch.cat(
            (all_latitude_longitude_embeddings, inputs[:, 2:]), dim=1
        )
        final_inputs = final_inputs.clone().detach()
        outputs = outputs.clone().detach()
        return final_inputs, outputs


# ************************************* Test data pipeline ***************


class Final_Air_Quality_Dataset_pipeline_latlon_PM25_test:
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
        self.torch_data = AirQualityDataset_latlon_PM25(
            self.Training_data, self.config, flag="Test"
        )
        return self.torch_data

    def full_pipeline(self):
        torch_latlong_loader = DataLoader(
            self.torch_data, batch_size=len(self.torch_data)
        )
        for inputs, outputs in torch_latlong_loader:
            all_latlong = inputs[:, 0:2]
        with torch.no_grad():
            all_latitude_longitude_embeddings = self.geospatial_encoder_instance(
                all_latlong
            )
        final_inputs = torch.cat(
            (all_latitude_longitude_embeddings, inputs[:, 2:]), dim=1
        )
        final_inputs = final_inputs.clone().detach()
        outputs = outputs.clone().detach()
        return final_inputs, outputs


# ---------------------------------------------------------------------------


# ************************************ Train data pipeline ********************************
class Final_Air_Quality_Dataset_pipeline_latlon:
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
        self.torch_data = AirQualityDataset_latlon(self.Training_data, self.config)
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


# ****************************** Validation data pipeline ******************************


class Final_Air_Quality_Dataset_pipeline_latlon_val:
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
        self.torch_data = AirQualityDataset_latlon(
            self.Training_data, self.config, flag="Val"
        )
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


# ***************************************** Test data pipeline ************************


class Final_Air_Quality_Dataset_pipeline_latlon_test:
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
        self.torch_data = AirQualityDataset_latlon(
            self.Training_data, self.config, flag="Test"
        )
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
