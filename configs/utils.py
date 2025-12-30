import yaml
import pandas as pd
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)


folder_path = (
    "/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Master-Thesis/configs"
)
yaml_path = os.path.join(folder_path, "config.yaml")

with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

print(config["experiments"])
