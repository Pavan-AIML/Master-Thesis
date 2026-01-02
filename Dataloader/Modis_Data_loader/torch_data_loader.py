# importing the required packages.
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import arrow  # to get time we use arrow.
import yaml


import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)
from configs.utils import config

# importing packages from the earlier file.
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import Modis_data_loader
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import PM_25_dataloader
from Dataloader.Modis_Data_loader.PM25_data_loader_analysis import (
    Training_data_loader,
)
from locationencoder.final_location_encoder import Geospatial_Encoder

# geospatial_encoder = Geospatial_Encoder(
#     config["Geo_spatial_Encoder"]["dim_in"],
#     config["Geo_spatial_Encoder"]["dim_hidden"],
#     config["Geo_spatial_Encoder"]["dim_out"],
#     config["Geo_spatial_Encoder"]["num_layers"],
# )

# here we will be building the class for getting our data in batches using torch data loader.
# in the torch we need to define 2 items clearly __len__(), __getitem__()
# config["output_vars_1"]# defining the output variable here.


# Here we will make the optimized code that will be used to replace all the classes those are lying extra here.


"""
Creating the common data loader classes for different data sets.
"""


class Airqualitydataset(Dataset):
    def __init__(self, df, config, flag="Train", input_type=1, output_type=1):
        # input types can be 1,2,3
        # output_types can be 1,2,3

        self.df = df
        self.config = config
        self.input_type = input_type
        self.output_type = output_type
        self.flag = flag
        self.input_cols = self.config["experiments"]["input_vars"][self.input_type]
        self.output_cols = self.config["experiments"]["output_vars"][self.output_type]
        config_ds_format = self.config["dataset"][
            self.config["experiments"]["dataset_num_1"]
        ]

        if flag == "Train":
            a, b = "train_start", "train_end"
        elif flag == "Val":
            a, b = "val_start", "val_end"
        elif flag == "Test":
            a, b = "test_start", "test_end"
        else:
            raise ValueError("flag is not matching with Train, Val, Test")

        self.start_time = self._get_time(config_ds_format[a])
        self.end_time = self._get_time(config_ds_format[b])

        self.df["date"] = pd.to_datetime(self.df["date"])
        self.data = self.df[
            (self.df["date"] >= self.start_time) & (self.df["date"] <= self.end_time)
        ]
        self.mean_AOD = self.data["AOD"].mean()
        self.std_AOD = self.data["AOD"].std() + 1e-8
        self.mean_PM25 = self.data["PM2.5"].mean()
        self.std_PM25 = self.data["PM2.5"].std() + 1e-8

        self.idx_input_AOD = self._get_col_index(self.input_cols, "AOD")
        self.idx_input_PM25 = self._get_col_index(self.input_cols, "PM2.5")
        self.idx_output_AOD = self._get_col_index(self.output_cols, "AOD")
        self.idx_output_PM25 = self._get_col_index(self.output_cols, "PM2.5")

    def _get_col_index(self, col_list, target_name):
        try:
            return col_list.index(target_name)
        except ValueError:
            return None

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1]).naive
        return arrow_time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_rows = self.data.iloc[index]
        x = torch.tensor(
            pd.to_numeric(data_rows[self.input_cols], errors="coerce")
            .astype(np.float32)
            .values,
            dtype=torch.float32,
        )
        y = torch.tensor(
            pd.to_numeric(data_rows[self.output_cols], errors="coerce")
            .astype(np.float32)
            .values,
            dtype=torch.float32,
        )
        
    # Now here we have normalized the AOD and PM2.5 values and have made them dynamics if the input and output columns exists in the given data set.
    
        if self.idx_input_AOD is not None:
            x[self.idx_input_AOD] = (
                x[self.idx_input_AOD] - self.mean_AOD
            ) / self.std_AOD
        if self.idx_input_PM25 is not None:
            x[self.idx_input_PM25] = (
                x[self.idx_input_PM25] - self.mean_PM25
            ) / self.std_PM25
        if self.idx_output_AOD is not None:
            y[self.idx_output_AOD] = (
                y[self.idx_output_AOD] - self.mean_AOD
            ) / self.std_AOD
        if self.idx_output_PM25 is not None:
            y[self.idx_output_PM25] = (
                y[self.idx_output_PM25] - self.mean_PM25
            ) / self.std_PM25

        return x, y


# class AirQualityDataset_latlon_AOD_PM25(Dataset):
#     # here flag will be default Train but if we assign something new it will consider that
#     # here the df will be the df coming out from our final trainig data from PM25_data_loader_analysis file

#     def __init__(self, df, config, flag="Train"):
#         # here we will store them as attributes.
#         if df is None or config is None:
#             raise ValueError("put the input data set and config file also")
#         self.df = df
#         self.config = config
#         self.flag = flag
#         # self.geospatial_encoder = Geospatial_Encoder(
#         #     config["Geo_spatial_Encoder"]["dim_in"],
#         #     config["Geo_spatial_Encoder"]["dim_hidden"],
#         #     config["Geo_spatial_Encoder"]["dim_out"],
#         #     config["Geo_spatial_Encoder"]["num_layers"],
#         # )
#         # Here as we are taking the data from config file the number of data sets it will make our code to make it moduler to also accept more than one data set if we have in future.

#         self.dataset_number = self.config["experiments"]["dataset_num_1"]

#         # This we use to extract the training, test and validation data sets dates in the given config file.

#         # this value is giving all the train and test values as per the flag.

#         ds_cfg = self.config["dataset"][self.dataset_number]
#         # Select input and target in such as way that we can change it later to do the changes easily.

#         # here we have selected the input type = 1

#         input_type_1 = self.config["experiments"]["input_type_1"]

#         # these are the input columns for the model
#         self.input_cols = self.config["experiments"]["input_vars"][input_type_1]

#         # these are the targets
#         self.target_cols = self.config["output_vars_1"]

#         # here we will select the range of the train, test and validation of data sets.
#         # select date range
#         if flag == "Train":
#             a, b = "train_start", "train_end"
#         elif flag == "Val":
#             a, b = "val_start", "val_end"
#         elif flag == "Test":
#             a, b = "test_start", "test_end"
#         else:
#             raise ValueError("flag is not matching with Train, Val, Test")
#         # let's define the functions
#         self.start_time = self._get_time(ds_cfg[a])
#         self.end_time = self._get_time(ds_cfg[b])
#         self.df["date"] = pd.to_datetime(self.df["date"])
#         self.data = self.df[
#             (self.df["date"] >= self.start_time) & (self.df["date"] <= self.end_time)
#         ]

#         self.mean_AOD = self.data["AOD"].mean()
#         self.std_AOD = self.data["AOD"].std() + 1e-8
#         self.mean_PM25 = self.data["PM2.5"].mean()
#         self.std_PM25 = self.data["PM2.5"].std() + 1e-8

#     def _get_time(self, time_yaml):
#         # in this line the data set that will be passed will be
#         arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1]).naive
#         return arrow_time

#     # get the length of the data-set
#     def __len__(self):
#         return len(self.data)

#     def print_data_type(self, idx):
#         data_rows = self.data.iloc[idx]
#         print(data_rows[self.input_cols].values)
#         print(data_rows[self.target_cols].values)

#     # get the items train as well as test from the data-set
#     def __getitem__(self, idx):
#         data_rows = self.data.iloc[idx]

#         x = torch.tensor(
#             # all the numeric values should match the data type all should be similer data type
#             pd.to_numeric(data_rows[self.input_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )

#         # all the numeric values should match the data type all should be similer data type
#         y = torch.tensor(
#             pd.to_numeric(data_rows[self.target_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         # ['latitude', 'longitude','AOD', 'PM2.5']
#         x[2] = (x[2] - self.mean_AOD) / self.std_AOD
#         x[3] = (x[3] - self.mean_PM25) / self.std_PM25
#         y[0] = (y[0] - self.mean_AOD) / self.std_AOD
#         y[1] = (y[1] - self.mean_PM25) / self.std_PM25
#         return x, y


# class AirQualityDataset_latlon_AOD(Dataset):
#     # here flag will be default Train but if we assign something new it will consider that
#     # here the df will be the df coming out from our final trainig data from PM25_data_loader_analysis file

#     def __init__(self, df, config, flag="Train"):
#         # here we will store them as attributes.
#         if df is None or config is None:
#             raise ValueError("put the input data set and config file also")
#         self.df = df
#         self.config = config
#         self.flag = flag

#         # Here as we are taking the data from config file the number of data sets it will make our code to make it moduler to also accept more than one data set if we have in future.

#         self.dataset_number = self.config["experiments"]["dataset_num_1"]

#         # This we use to extract the training, test and validation data sets dates in the given config file.

#         ds_cfg = self.config["dataset"][self.dataset_number]
#         # Select input and target in such as way that we can change it later to do the changes easily.

#         # here we have selected the input type = 1

#         input_type_2 = self.config["experiments"]["input_type_2"]

#         # these are the input columns for the model
#         self.input_cols = self.config["experiments"]["input_vars"][input_type_2]
#         # these are the targets
#         self.target_cols = self.config["output_vars_1"]

#         # here we will select the range of the train, test and validation of data sets.
#         # select date range
#         if flag == "Train":
#             a, b = "train_start", "train_end"
#         elif flag == "Val":
#             a, b = "val_start", "val_end"
#         elif flag == "Test":
#             a, b = "test_start", "test_end"
#         else:
#             raise ValueError("flag is not matching with Train, Val, Test")
#         # let's define the functions
#         self.start_time = self._get_time(ds_cfg[a])
#         self.end_time = self._get_time(ds_cfg[b])
#         self.df["date"] = pd.to_datetime(self.df["date"])
#         self.data = self.df[
#             (self.df["date"] >= self.start_time) & (self.df["date"] <= self.end_time)
#         ]

#     def _get_time(self, time_yaml):
#         # in this line the data set that will be passed will be
#         arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1]).naive
#         return arrow_time

#     # get the length of the data-set
#     def __len__(self):
#         return len(self.data)

#     def print_data_type(self, idx):
#         data_rows = self.data.iloc[idx]
#         print(data_rows[self.input_cols].values)
#         print(data_rows[self.target_cols].values)

#     # get the items train as well as test from the data-set
#     def __getitem__(self, idx):
#         data_rows = self.data.iloc[idx]

#         x = torch.tensor(
#             # all the numeric values should match the data type all should be similer data type
#             pd.to_numeric(data_rows[self.input_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         # all the numeric values should match the data type all should be similer data type
#         y = torch.tensor(
#             pd.to_numeric(data_rows[self.target_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         return x, y


# # -------------************---------------------------------
# # Data set with inputs latitude, longitude and AOD values as inputs.
# # Data set with AOD and PM2.5 values as inputs.


# class AirQualityDataset_latlon_PM25(Dataset):
#     # here flag will be default Train but if we assign something new it will consider that
#     # here the df will be the df coming out from our final trainig data from PM25_data_loader_analysis file

#     def __init__(self, df, config, flag="Train"):
#         # here we will store them as attributes.
#         if df is None or config is None:
#             raise ValueError("put the input data set and config file also")
#         self.df = df
#         self.config = config
#         self.flag = flag

#         # Here as we are taking the data from config file the number of data sets it will make our code to make it moduler to also accept more than one data set if we have in future.

#         self.dataset_number = self.config["experiments"]["dataset_num_1"]

#         # This we use to extract the training, test and validation data sets dates in the given config file.

#         ds_cfg = self.config["dataset"][self.dataset_number]
#         # Select input and target in such as way that we can change it later to do the changes easily.

#         # here we have selected the input type = 1

#         input_type_3 = self.config["experiments"]["input_type_3"]

#         # these are the input columns for the model
#         self.input_cols = self.config["experiments"]["input_vars"][input_type_3]
#         # these are the targets
#         self.target_cols = self.config["output_vars_1"]

#         # here we will select the range of the train, test and validation of data sets.
#         # select date range
#         if flag == "Train":
#             a, b = "train_start", "train_end"
#         elif flag == "Val":
#             a, b = "val_start", "val_end"
#         elif flag == "Test":
#             a, b = "test_start", "test_end"
#         else:
#             raise ValueError("flag is not matching with Train, Val, Test")
#         # let's define the functions
#         self.start_time = self._get_time(ds_cfg[a])
#         self.end_time = self._get_time(ds_cfg[b])
#         self.df["date"] = pd.to_datetime(self.df["date"])
#         self.data = self.df[
#             (self.df["date"] >= self.start_time) & (self.df["date"] <= self.end_time)
#         ]

#     def _get_time(self, time_yaml):
#         # in this line the data set that will be passed will be
#         arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1]).naive
#         return arrow_time

#     # get the length of the data-set
#     def __len__(self):
#         return len(self.data)

#     def print_data_type(self, idx):
#         data_rows = self.data.iloc[idx]
#         print(data_rows[self.input_cols].values)
#         print(data_rows[self.target_cols].values)

#     # get the items train as well as test from the data-set
#     def __getitem__(self, idx):
#         data_rows = self.data.iloc[idx]

#         x = torch.tensor(
#             # all the numeric values should match the data type all should be similer data type
#             pd.to_numeric(data_rows[self.input_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         # all the numeric values should match the data type all should be similer data type
#         y = torch.tensor(
#             pd.to_numeric(data_rows[self.target_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         return x, y


# class AirQualityDataset_latlon(Dataset):
#     # here flag will be default Train but if we assign something new it will consider that
#     # here the df will be the df coming out from our final trainig data from PM25_data_loader_analysis file

#     def __init__(self, df, config, flag="Train"):
#         # here we will store them as attributes.
#         if df is None or config is None:
#             raise ValueError("put the input data set and config file also")
#         self.df = df
#         self.config = config
#         self.flag = flag

#         # Here as we are taking the data from config file the number of data sets it will make our code to make it moduler to also accept more than one data set if we have in future.

#         self.dataset_number = self.config["experiments"]["dataset_num_1"]

#         # This we use to extract the training, test and validation data sets dates in the given config file.

#         ds_cfg = self.config["dataset"][self.dataset_number]
#         # Select input and target in such as way that we can change it later to do the changes easily.

#         # here we have selected the input type = 1

#         input_type_4 = self.config["experiments"]["input_type_4"]

#         # these are the input columns for the model
#         self.input_cols = self.config["experiments"]["input_vars"][input_type_4]
#         # these are the targets
#         self.target_cols = self.config["output_vars_1"]

#         # here we will select the range of the train, test and validation of data sets.
#         # select date range
#         if flag == "Train":
#             a, b = "train_start", "train_end"
#         elif flag == "Val":
#             a, b = "val_start", "val_end"
#         elif flag == "Test":
#             a, b = "test_start", "test_end"
#         else:
#             raise ValueError("flag is not matching with Train, Val, Test")
#         # let's define the functions
#         self.start_time = self._get_time(ds_cfg[a])
#         self.end_time = self._get_time(ds_cfg[b])
#         self.df["date"] = pd.to_datetime(self.df["date"])
#         self.data = self.df[
#             (self.df["date"] >= self.start_time) & (self.df["date"] <= self.end_time)
#         ]

#     def _get_time(self, time_yaml):
#         # in this line the data set that will be passed will be
#         arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1]).naive
#         return arrow_time

#     # get the length of the data-set
#     def __len__(self):
#         return len(self.data)

#     def print_data_type(self, idx):
#         data_rows = self.data.iloc[idx]
#         print(data_rows[self.input_cols].values)
#         print(data_rows[self.target_cols].values)

#     # get the items train as well as test from the data-set
#     def __getitem__(self, idx):
#         data_rows = self.data.iloc[idx]

#         x = torch.tensor(
#             # all the numeric values should match the data type all should be similer data type
#             pd.to_numeric(data_rows[self.input_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         # all the numeric values should match the data type all should be similer data type
#         y = torch.tensor(
#             pd.to_numeric(data_rows[self.target_cols], errors="coerce")
#             .astype(np.float32)
#             .values,
#             dtype=torch.float32,
#         )
#         return x, y


# import torch
# import pandas as pd

# df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
# df.columns = ["lat", "lon", "AOD", "PM25"]
# df
# tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# x, y = tensor
# x[2] = 9
# y[2] = 10


# df_1 = df.iloc[0:2]
# # df_1[3] =df_1[3]/2
# df_1.iloc[0]
