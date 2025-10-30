import yaml
import pandas as pd
import os 


folder_path = "/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Master-Thesis/notebooks/Modis_data_analysis"
yaml_path = os.path.join(folder_path, "config.yaml")

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
print(config["experiments"])