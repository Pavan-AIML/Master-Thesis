# with the help of this file we can import the modules from Thesis folder

import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[2] / "locationencoder"))
# s
# ys.path.append("../../Master-Thesis/")
"""
# Making the Master thesis folder as root folder for imports 

"""
ROOT = Path(__file__).resolve().parents[2]  # tests/ -> project root
sys.path.insert(0, str(ROOT))
print(ROOT)


from locationencoder.final_location_encoder import Geospatial_Encoder


"""
Making instance of the Geospatial_Encoder
"""

geospatial_encoder_instance = Geospatial_Encoder()