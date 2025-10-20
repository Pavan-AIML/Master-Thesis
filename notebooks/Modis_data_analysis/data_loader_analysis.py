import pandas as pd
import os
import numpy as np
from scipy.spatial import cKDTree


class Modis_data_loader:
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


# In case we do not want to add any other methods or attributes in this case we can use the below code that is how we write a child class.

"""

class PM2_5DataAnalysis(DataLoaderAnalysis):
    def __init__(
        self,
        file_name,
        PM_25_file_name,
        folder_root_dir="/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Thesis/src/data/data_files",
    ):
        super().__init__(file_name, folder_root_dir)

        full_PM_path = os.path.join(folder_root_dir, PM_25_file_name)

        self.PM_data = pd.DataFrame(np.load(full_PM_path, allow_pickle=True))

    def get_PM_head(self, n=20):
        return self.PM_data.head(n)

    def get_PM_shape(self):
        return self.PM_data.shape
        
"""

"""
As this file is a .npy file with 3D data hence we need to build a dataloader that can take the 3D data and can extract the files in steps.  

"""


class PM_25_dataloader:
    def __init__(
        self,
        filename="PM_2.5_Data/3_hr_data_Sept_2023_v1.npy",
        folder_root_dir="/Users/pavankumar/Documents/Winter_Thesis/Coding_Learning/Thesis/src/data/data_files",
    ):
        self.filename = filename
        self.folder_root_dir = folder_root_dir
        self.full_path = os.path.join(self.folder_root_dir, self.filename)
        self.data = np.load(self.full_path, allow_pickle=True)

    def get_shape(self):
        if self.data is not None:
            return self.data.T.shape
        else:
            raise ValueError("Data not loaded properly.")

    def get_data(self):
        if self.data is not None:
            data = np.load(self.full_path, allow_pickle=True)
            data = np.float32(data)
            data = data[:, :, 0]
            data = pd.DataFrame(data)
            data = data.T
            return data
        else:
            raise ValueError("Data not loaded properly.")


def combine_the_data_frames(df_1, df_2):
    if df_1 is not None and df_2 is not None:
        df_final = pd.concat([df_1, df_2], axis=1)
        return df_final
    else:
        print("check the data frames")


class Training_data_loader:
    def __init__(self, AOD_data_1, Aod_data_2, AOD_data_3, Pm_data, stn_data):
        self.AOD_data_1 = AOD_data_1
        self.Aod_data_2 = Aod_data_2
        self.AOD_data_3 = AOD_data_3
        self.Pm_data = Pm_data
        self.stn_data = stn_data

    # Step 1 we will fuse the AOD data

    def AOD_data_fusion(self):
        AOD_data_1 = self.AOD_data_1
        AOD_data_2 = self.Aod_data_2.drop(["latitude", "longitude"], axis=1)
        AOD_data_3 = self.AOD_data_3.drop(["latitude", "longitude"], axis=1)
        AOD_final = pd.concat([AOD_data_1, AOD_data_2, AOD_data_3], axis=1)
        return AOD_final

    # second step is to find the nearest neighbours of the PM2.5 data points from the AOD data points.

    def PM25_nearest_neighbour_finder(self, AOD_final):
        stn_cordinates = self.stn_data[["latutude", "longitude"]].values

        AOD_cordinates = self.AOD_data_1[["latitude", "longitude"]].values

        # creating the ckd tree of the AOD cordinates.
        CKD_tree = cKDTree(AOD_cordinates)

        # Now we will find the nearest    neighbours AOD data points for the PM2.5 data points.
        dist, indx = CKD_tree.query(stn_cordinates, k=1)
        nearest_neighbours = AOD_final.iloc[indx].reset_index(drop=True)
        nearest_neighbours = pd.DataFrame(nearest_neighbours)
        return nearest_neighbours

    def data_cleaning_and_missing_values_handeling(self, nearest_neighbours):
        # Handling duplicated columns by averaging their values
        duplicated_columns = nearest_neighbours.columns[
            nearest_neighbours.columns.duplicated()
        ].unique()

        for col in duplicated_columns:
            cols_to_average = nearest_neighbours.loc[:, col]
            averaged_col = cols_to_average.mean(axis=1)
            nearest_neighbours[col] = averaged_col

        nearest_neighbours = nearest_neighbours.loc[
            :, ~nearest_neighbours.columns.duplicated()
        ]
