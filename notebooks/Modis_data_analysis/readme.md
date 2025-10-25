# In this note book I have explained all the steps taken to create the dat aloader using object oriented programing. 

```js

import pandas as pd
import os


class DataLoaderAnalysis:
    def __init__(
        self,
        file_name,
        folder_root_dir="/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Thesis/src/data/data_files",
    ):
        """
        Initialize the DataLoaderAnalysis with a file name and root directory.

        Args:
            file_name (str): Name of the file to load.
            folder_root_dir (str): Root directory path for data files (default: MODIS_AOD path).

        Raises:
            ValueError: If file_name is empty or not a string.
            FileNotFoundError: If the file does not exist.
            Exception: For other file reading errors.
        """
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("file_name must be a non-empty string")

        self.folder_root_dir = folder_root_dir
        self.file_name = file_name
        self.full_path = os.path.join(folder_root_dir, file_name)

        try:
            self.data = pd.read_csv(self.full_path)  # Modify if not CSV
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.full_path} not found")
        except Exception as e:
            raise Exception(f"Error loading file {self.full_path}: {str(e)}")

    def get_head(self, n=20):
        """
        Return the first n rows of the data.

        Args:
            n (int): Number of rows to display (default: 5).

        Returns:
            pandas.DataFrame: First n rows of the data.
        """
        return self.data.head(n)

    def get_shape(self):
        """
        Return the shape of the data (rows, columns).

        Returns:
            tuple: Shape of the data (rows, columns).
        """
        return self.data.shape

```

## Now we will see how can we use the inheritance to inherit the attributes from the above class. 

## From this we can inherit all the attributes of the parent class DataLoaderAnalysis we can do like below.

```js
class PM2_5DataAnalysis(DataLoaderAnalysis):
    pass
    
```

## If we want to add more attributes in the class other than parent class then we need to add the super().__int__()

## So from here we will use above class to do the inheritance.

```js 

class PM2_5DataAnalysis(DataLoaderAnalysis):
    def __init__(self, filename, folder_root_dir, PM_25_file_name):
    

# we need to inherit the attributes from the parent class. 

    super().__init__(filename, folder_root_dir)

    self.PM_25_file_name = PM_25_file_name

    full_path = os.path.join(folder_root_dir, PM_25_file_name)

    def PM_25_get_head(self):
       npy_data = np.load(full_path, allow_pickle = True)

       # convert ot pandas data frame 
       df = pd.DataFrame(npy_data)
       return df.head(10)
    
# Creating the instances 

   PM2_5DataAnalysis()

```

### From here as you can see that the shape of the PM2.5 is `8760, 40` that means from 40 stations the values of the PM_2.5 are coming every hour. So the `data PM2.5` `locations = 40` and `time stamps = 8760` 


### Now we see that the shape of the AOD data is `1350, 365`  hence the `data AOD` `locations = 1350` and `time_stamps = 365` 


## Input data types ?

### In this case the input data features will be the following. 

```js
Inputs 

1> Embeddings of [ Lat , Long, PM2.5, AOD ]

2 > Embeddings of [ Lat, Long, PM2.5 ]

3 > Embeddings of [ Lat, Long, AOD ]

Outputs 

target values of [ Pm2.5, AOD ] 

```

### In this we see that the `spatial dim of AOD data is bigger than PM2.5 data` and `temporal dim of PM2.5 data is biggern that AOD` data.


## Dimension matching `Options`.

```js
For temporal_dimension: 

Either we can reduce the temporal dimension of PM2.5 data to daily by taking the average of the PM2.5 data.

or 

Interpolate the AOD data for hourly to match the temporal dimension.


For spatial_dimension:

Building a NN Tree in which the center nodes will be the PM2.5 nodes and surrounding nodes will be AOD nodes. 


```

## KD Tree :

K : dimension of the space where the dataset has been organized.


## Fuseed data frame creation 

```js


```

## Final data frame creation.

```js
def get_final_training_data(self, nearest_neighbours, daily_PM_data):
    results = []
    # Ensure date columns are datetime in both DataFrames
    date_columns = nearest_neighbours.columns[2:]
    nearest_neighbours = nearest_neighbours.copy()  # Avoid modifying original
    nearest_neighbours.columns = ["latitude", "longitude"] + [
        pd.to_datetime(col) for col in date_columns
    ]
    # Ensure daily_PM_data columns are datetime
    daily_PM_data = daily_PM_data.copy()
    daily_PM_data.columns = [pd.to_datetime(col) for col in daily_PM_data.columns]
    # Verify index alignment
    if not nearest_neighbours.index.equals(daily_PM_data.index):
        # Align indices (assuming same locations, different order)
        daily_PM_data = daily_PM_data.reindex(nearest_neighbours.index)
        if daily_PM_data.isna().all().all():
            raise ValueError(
                "Index alignment failed: No matching indices between nearest_neighbours and daily_PM_data"
            )
    for index, row in nearest_neighbours.iterrows():
        latitude = row["latitude"]
        longitude = row["longitude"]
        # Iterate over date columns, not row.index
        for date_col in nearest_neighbours.columns[2:]:
            aod_val = row[date_col]
            if pd.isna(aod_val):
                continue
            try:
                # Use loc for label-based indexing to handle non-integer indices
                pm_val = daily_PM_data.loc[index, date_col]
            except (KeyError, IndexError):
                continue
            if pd.isna(pm_val):
                continue
            results.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "date": date_col,
                    "MODIS_AOD": aod_val,
                    "PM25": pm_val,
                }
            )
    final_training_data = pd.DataFrame(results)
    if final_training_data.empty:
        print(
            "Warning: No valid data points found. Check for missing values or index/column mismatches."
        )
    return final_training_data


    size of nearest_neighbours > (40, 1097 ) with index
    size of daily PM data > (40, 365) with index 

```