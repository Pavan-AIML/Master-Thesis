# In this note book I have just tried to show how the inheritance works ?

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