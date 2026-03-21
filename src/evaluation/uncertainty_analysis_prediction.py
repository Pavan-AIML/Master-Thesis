import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
run_name = f"Eval_Run_{current_time}"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set-up path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Local packages imports.
from configs.utils import config
from locationencoder.final_location_encoder import Geospatial_Encoder
from Dataloader.Modis_Data_loader.final_loader import Final_Air_Quality_Dataset_pipeline
from src.Models.neural_process import NeuralProcess
from src.Models.neural_process import Robust_Neural_Process
from src.training.optimizer_utils import neural_process_data
from test_model_utils import NP_Evaluator
# from src.evaluation.test_model_utils import plot_measured_vs_estimated
SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
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
    plot_dir = ROOT / "final_plots" / run_name

    os.makedirs(plot_dir, exist_ok=True)

    eval_input_types = [8, 10,11, 12]
    base_input_for_eval = {8:8 , 10:10, 11: 11, 12:12}
    
    output_type = [1, 2, 3]

    # I have stored the checkpoints in the form of input points and output points, hence I need to
    for eval_i in eval_input_types:
        # Here we will construct the experiment name.
        for j in output_type:
            eval_in_columns = config["experiments"]["input_vars"][eval_i]
            out_columns = config["experiments"]["output_vars"][j]
            exp_name = f"{'_'.join(eval_in_columns)}____{'_'.join(out_columns)}"

            # Find the base directory
            # this directory will land us to the model_type directory inside the cpu_checkpoints.
            base_i = eval_i
            base_in_columns= config["experiments"]["input_vars"][base_i]
            base_exp_name = f"{'_'.join(base_in_columns)}____{'_'.join(out_columns)}"
            base_exp_dir = ROOT / "Gpu_checkpoints_tum" / base_exp_name

            # Varification check if this experiment exist.

            if not os.path.exists(base_exp_dir):
                print(f"Skippking {exp_name}(folder not found)")

                continue
            # After finding the experiment name I will search for the latest timestamp folder.

            # all_time_stamps = glob.glob(os.path.join(base_exp_dir, "*"))

            all_time_stamps = [
                                p for p in glob.glob(os.path.join(base_exp_dir, "*"))
                                if os.path.isdir(p)
            ]

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
                base_pipeline = Final_Air_Quality_Dataset_pipeline(
                config, geospatial_encoder, input_type=base_i, output_type=j, flag="Train"
                )
                base_pipeline.full_pipeline()
                base_stats = base_pipeline.torch_data.data
                pipeline = Final_Air_Quality_Dataset_pipeline(
                    config, geospatial_encoder, input_type=eval_i, output_type=j, flag="Val"
                )
                val_x, val_y = pipeline.full_pipeline(train_stats_override=base_stats)
                stats_source = pipeline.torch_data

            except Exception as e:
                print(f"[SKIP] {exp_name}: Data load error: {e}")
                traceback.print_exc()
                continue
            ds = neural_process_data(val_x, val_y, 25)
            dl = DataLoader(ds, batch_size=20, shuffle=False)

            x_dim, y_dim = val_x.shape[-1], val_y.shape[-1]
            # making the instance of model class. 
            model = NeuralProcess(x_dim, y_dim, 128, 128)

            evaluator = NP_Evaluator(model, device)
            try:
                # Load weights in the model.
                print(f"Loading the checkpoints {best_checkpoint}")
                evaluator.load_best_checkpoints(base_exp_dir)

            except Exception as e:
                print(f"[SKIP] {exp_name}: Model load error: {e}")
                traceback.print_exc()
                continue

            # Here we will define the model.

            y_true, mu_y, y_nll, coverage_score = evaluator.run_eval(
                dl, out_columns, plot_dir, exp_name
            )

            # Metric and saving

            for idx, var_name in enumerate(out_columns):
                
                flat_t = y_true[:, :, idx]
                flat_p = mu_y[:, :, idx]
                
                # Fix the values of the mean nll
                
                if len(y_nll.shape) > 1 :
                    nll_mean = np.mean(y_nll[:,idx])
                else:
                    nll_mean = np.mean(y_nll)
                
                if isinstance(coverage_score, (list, np.ndarray)) and len(coverage_score) == len(out_columns):
                    current_coverage = coverage_score[idx]
                else:
                    current_coverage = coverage_score
                
                # Here we try to get the un normalize data. 
                if "PM2.5" in var_name:
                    mean_val = stats_source.mean_PM25
                    std_val = stats_source.std_PM25
                    
                elif "AOD" in var_name:
                    mean_val = stats_source.mean_AOD
                    std_val = stats_source.std_AOD
                else:
                    mean_val, std_val = 0.0, 1.0
                    
                    
                # NLL is already averaged per sample in run_eval, so we take mean of all samples
                real_t = flat_t * std_val + mean_val
                real_p = flat_p * std_val + mean_val
                rmse = np.sqrt(mean_squared_error(real_t.flatten(), real_p.flatten()))
                r2 = r2_score(real_t.flatten(), real_p.flatten())
                #  # Measured vs Estimated scatter with R2 and fit
                # plot_measured_vs_estimated(real_t, real_p, var_name, exp_name, plot_dir)

                # Sanity checks 
                if idx ==0:
                    print(f"\n [Sanity CHECK for {var_name}]")
                    print(f"Normalization Factor: Mean = {mean_val:.4f}, STd = {std_val:.4f}")
                    print(f"  Raw Model Output Sample: {flat_p[0,0]:.4f}")
                    print(f"  Un-normalized Pred Sample: {real_p[0,0]:.4f}")
                    print(f"  Un-normalized Truth Sample: {real_t[0,0]:.4f}")

    # Check for "Flatline" prediction
                if np.abs(real_p.max() - real_p.min()) < 1e-4:
                    print("  WARNING: Model is predicting the SAME value for everything (Mean Collapse)!")
            
                rmse = np.sqrt(mean_squared_error(real_t.flatten(), real_p.flatten()))
                r2 = r2_score(real_t.flatten(), real_p.flatten())
                print(
                    f" -> {var_name} | RMSE : {rmse:.4f} |R2 {r2:.4f} | NLL: {nll_mean: .4f} "
                )

                results.append(
                    {
                        "Experiment ": exp_name,
                        "Variable  ": var_name,
                        "RMSE": rmse,
                        "R2": r2,
                        "NLL": nll_mean,
                        "95 %_Coverage":current_coverage
                    }
                )
    if results:
        results_dir = ROOT / "GPU_final_results" / run_name

        os.makedirs(results_dir, exist_ok = True)
        
        csv_path = results_dir / "final_results_summery.csv"
        # make a new directory or replace it with the existing one.
        os.makedirs(results_dir, exist_ok=True)
        # Save the file
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"The directory has been created {results_dir}")
        print(f"Saved final results in {csv_path}")


if __name__ == "__main__":
    Neural_process_run_evaluation()
