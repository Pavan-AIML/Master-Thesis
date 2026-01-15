import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# Set-up path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Local packages imports.
from configs.utils import config
from locationencoder.final_location_encoder import Geospatial_Encoder
from Dataloader.Modis_Data_loader.final_loader import Final_Air_Quality_Dataset_pipeline
from src.Models.neural_process import NeuralProcess
from src.training.optimizer_utils import neural_process_data
from test_model_utils import NP_Evaluator


# Below we will build the uncertainty analysis final folder that will give us the visual picture of the predicted and the real points.


def Neural_process_run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Here we can check also which device we are using
    print(f"Using device {device}")

    # In Encoder the shared across experiments.

    #   Collecting the data from
    #   dim_in: 100
    #   dim_hidden: 128
    #   dim_out: 126
    #   num_layers: 6

    geospatial_encoder = Geospatial_Encoder(
        config["Geo_spatial_Encoder"]["dim_in"],
        config["Geo_spatial_Encoder"]["dim_hidden"],
        config["Geo_spatial_Encoder"]["dim_out"],
        config["Geo_spatial_Encoder"]["num_layers"],
    )
    # I have stired the results here
    results = []

    input_type = [1, 2, 3, 4]
    output_type = [1, 2, 3]
    # I have stored the checkpoints in the form of input points and output points, hence I need to
    for i in input_type:
        # Here we will construct the experiment name.
        for j in output_type:
            in_columns = config["experiments"]["input_vars"][i]
            out_columns = config["experiments"]["input_vars"][j]
            exp_name = f"{'_'.join(in_columns)}____{'_'.join(out_columns)}"

            # Find the base directory
            # this directory will land us to the model_type directory inside the cpu_checkpoints.

            base_exp_dir = ROOT / "cpu_checkpoints" / exp_name

            # Varification check if this experiment exist.

            if not os.path.exists(base_exp_dir):
                print(f"Skippking {exp_name}(folder not found)")

                continue
            # After finding the experiment name I will search for the latest timestamp folder.

            all_time_stamps = glob.glob(os.path.join(base_exp_dir, "*"))

            if not all_time_stamps:
                print(f"Skipping {exp_name}: (No timestamp folder found.)")
                continue
            # here key refers to the timestamps that it needs to look.
            latest_timestamp_folder = max(all_time_stamps, key=os.path.getctime)

            print(f"processing : {exp_name}")
            print(f"Found latest run :{os.path.basename(latest_timestamp_folder)}")

            # Explore the check point files and find the best check point files.

            check_points_files = glob.glob(
                os.path.join(latest_timestamp_folder, "*.pth")
            )

            if not check_points_files:
                print(
                    f"No .pth files found in {os.path.basename(latest_timestamp_folder)}"
                )
                continue
            try:
                best_checkpoint = max(
                    check_points_files,
                    key=lambda f: int(f.split("_")[-1].split(".")[0]),
                )
            except ValueError:
                best_checkpoint = max(check_points_files, key=os.path.getctime)

            print(f"Processing {exp_name}")
            print(
                f"The choosen time stamp : {os.path.basename(os.path.basename(latest_timestamp_folder))}"
            )
            print(f"Used check-point file : {os.path.basename(best_checkpoint)}")

            print(f"Loading the final air quality data")
            try:
                pipeline = Final_Air_Quality_Dataset_pipeline(
                    config, geospatial_encoder, i, j
                )
                train_x, train_y = pipeline.full_pipeline()

            except Exception as e:
                print(f"Data load error: {e}")
                continue
            ds = neural_process_data(train_x, train_y, 200)
            dl = DataLoader(ds, batch_size=20, shuffle=False)

            x_dim, y_dim = train_x.shape[-1], train_y.shape[-1]

            # Here we will define the model.

            print(f" >>>> Loading checkpoint: {os.path.basename(best_checkpoint)}")

            evaluator = NP_Evaluator(model, device)
