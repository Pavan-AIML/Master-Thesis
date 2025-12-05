import pandas as pd

nearest_neighbors = pd.DataFrame(
    {
        "latitude": [28.61, 28.52],
        "longitude": [77.23, 77.12],
        "2020-01-01": [0.45, 0.39],
        "2020-01-02": [0.48, None],
    }
)

print(nearest_neighbors)


# Testing the code to print each row's index and values

for ind, col in nearest_neighbors.iterrows():
    print(f"Row index: {ind}")
    print(f"Latitude: {col['latitude']}")
    print(f"longitude:{col['longitude']}")


# for idx, row in nearest_neighbors.iterrows():
#     print(f"Row index: {idx}")
#     print(f"Latitude: {row['latitude']}")
#     print(f"Longitude: {row['longitude']}")
#     print(f"AOD on 2020-01-01: {row['2020-01-01']}")
#     print(f"AOD on 2020-01-02: {row['2020-01-02']}")
#     print("-" * 30)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

data_rows_1 =torch.tensor(pd.to_numeric([1,2,3,5]))
data_rows_1.shape          
              
                        
data_rows_2 = torch.tensor(pd.to_numeric([5,6,7,8]))

combined_data = torch.cat((data_rows_1, data_rows_2), dim =-1 )

combined_data


data_rows_1.columns = ["lat", "lon","AOD", "PM2.5"]

data_rows_2.columns = ["lat_2", "lon_2", "AOD_2"]

