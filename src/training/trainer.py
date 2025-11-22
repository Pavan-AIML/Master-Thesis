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
from notebooks.Modis_data_analysis.torch_data_loader import AirQualityDataset

# importing config files


from notebooks.Modis_data_analysis.final_loader import (
    Final_Air_Quality_Dataset_pipeline,
)

instance = Final_Air_Quality_Dataset_pipeline(config)

instance.modis_data_sets()
instance.stations_data_sets()
instance.PM_25_data()
instance.training_data()
instance.Torch_data()
final_data = instance.full_pipeline()


# -------------************--------------------------------
"""
Final training data 
"""

final_data[0][0]
final_data[1][0]
