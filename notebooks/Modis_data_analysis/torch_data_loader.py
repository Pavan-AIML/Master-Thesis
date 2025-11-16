# importing the required packages.
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import arrow  # to get time we use arrow.
import yaml
from utils import config

# importing packages from the earlier file.
from PM25_data_loader_analysis import Modis_data_loader
from PM25_data_loader_analysis import PM_25_dataloader
from PM25_data_loader_analysis import Training_data_loader

# here we will be building the class for getting our data in batches using torch data loader.
# in the torch we need to define 2 items clearly __len__(), __getitem__()


class AirQualityDataset(Dataset):
    # here flag will be default Train but if we assign something new it will consider that
    # here the df will be the df coming out from our final trainig data from PM25_data_loader_analysis file

    def __init__(self, df, config, flag="Train"):
        # here we will store them as attributes.
        if df is None or config is None:
            raise ValueError("put the input data set and config file also")
        self.df = df
        self.config = config
        self.flag = flag

        # Here as we are taking the data from config file the number of data sets it will make our code to make it moduler to also accept more than one data set if we have in future.

        self.dataset_number = self.config["experiments"]["dataset_num_1"]

        # This we use to extract the training, test and validation data sets dates in the given config file.

        ds_cfg = self.config["dataset"][self.dataset_number]
        # Select input and target in such as way that we can change it later to do the changes easily.

        # here we have selected the input type = 1

        input_type_1 = self.config["experiments"]["input_type_1"]

        # these are the input columns for the model
        self.input_cols = self.config["experiments"]["input_vars"][input_type_1]
        # these are the targets
        self.target_cols = self.config["experiments"]["target_type"]
        # here we will select the range of the train, test and validation of data sets.
        # select date range
        if flag == "Train":
            a, b = "train_start", "train_end"
        elif flag == "Val":
            a, b = "val_start", "val_end"
        elif flag == "Test":
            a, b = "test_start", "test_end"
        else:
            raise ValueError("flag is not matching with Train, Val, Test")
        # let's define the functions
        self.start_time = self._get_time(ds_cfg[a])
        self.end_time = self._get_time(ds_cfg[b])
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.data = self.df[
            (self.df["date"] >= self.start_time) & (self.df["date"] <= self.end_time)
        ]

    def _get_time(self, time_yaml):
        # in this line the data set that will be passed will be
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1]).naive
        return arrow_time

    # get the length of the data-set
    def __len__(self):
        return len(self.data)

    def print_data_type(self, idx):
        data_rows = self.data.iloc[idx]
        print(data_rows[self.input_cols].values)
        print(data_rows[self.target_cols].values)

    # get the items train as well as test from the data-set
    def __getitem__(self, idx):
        data_rows = self.data.iloc[idx]

        x = torch.tensor(
            # all the numeric values should match the data type all should be similer data type
            pd.to_numeric(data_rows[self.input_cols], errors="coerce")
            .astype(np.float32)
            .values,
            dtype=torch.float32,
        )
        # all the numeric values should match the data type all should be similer data type
        y = torch.tensor(
            pd.to_numeric(data_rows[self.target_cols], errors="coerce")
            .astype(np.float32)
            .values,
            dtype=torch.float32,
        )
        return x, y
