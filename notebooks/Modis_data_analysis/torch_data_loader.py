# importing the required packages.

import torch
from torch.utils.data import dataset
from torch.utils.data import dataloader
import pandas as pd
import numpy as np
from datetime import datetime
import arrow  # to get time we use arrow.

# importing packages from the earlier file.

from PM25_data_loader_analysis import Modis_data_loader

from PM25_data_loader_analysis import PM_25_dataloader

from PM25_data_loader_analysis import Training_data_loader


# here we will be building the class for getting our data in batches using torch data loader.

# in the torch we need to define 2 items clearly __len__(), __getitem__()


class AirQualityDataset(dataset):
    # here flag will be default Train but if we assign something new it will consider that

    # here the df will be the df coming out from our final trainig data from PM25_data_loader_analysis file
    def __init__(self, df, config, flag="train"):
        # here we will store them as attributes.

        self.df = df
        self.config = config
        self.flag = flag
        # Here as we are taking the data from config file the number of data sets it will make our code to make it moduler to also accept more than one data set if we have in future.

        self.dataset_number = self.config["experiments"]["dataset_num"]
        ds_cfg = config["dataset"][self.dataset_number]

        # Select input and target in such as way that we can change it later to do the changes easily.

        input_type = config["experiments"]["input_type"]

        # these are the input columns for the model
        self.feature_cols = config["experiments"]["feature_sets"]

        # these are the targets

        self.target_cols = config["experiment"]["target_type"]

        # here we will select the range of the train, test and validation of data sets.

        # select date range
        if flag == "Train":
            a, b = "train_start", "train_end"

        if flag == "Val":
            a, b == "val_start", "val_end"

        if flag == "Test":
            a, b == "test_start", "test_end"

        else:
            raise ValueError("flag is not matching with Train, Val, Test")

    pass
