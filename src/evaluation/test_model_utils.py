# import all the necessary packages
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# os.path.dirname(__file__) >> Current file path we go one step above and then add the absolute path in the sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.optimizer_utils import context_target_split
from training.optimizer_utils import neural_process_data
from Models.neural_process import NeuralProcess
from configs.utils import config
from locationencoder.final_location_encoder import Geospatial_Encoder
from Dataloader.Modis_Data_loader.final_loader import Final_Air_Quality_Dataset_pipeline
from src.Models.neural_process import NeuralProcess
from src.Models.neural_process import Robust_Neural_Process
from training.optimizer_utils import neural_process_data, context_target_split
from tqdm import tqdm
from pathlib import Path
# Need to access context_target_split from your existing optimizer utils
# Make sure optimizer_utils is in the python path or adjust import
from training.optimizer_utils import context_target_split
# Setup ROOT for default plot directory
ROOT = Path(__file__).resolve().parents[2]
# from the below code we will visualize our predicted points.
def visualize_sample(
    xc,
    yc,
    xt,
    yt,
    mu_yt,
    var_yt,
    output_cols,
    experiment_name,
    sample_idx=0,
    save_dir=".plots",
):
    """
    Plots Context (Blue), Target (Red), Prediction Mean (Green Line), and Uncertainty (Green Shaded).
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. MOVE TO CPU/NUMPY & EXTRACT REAL DATA
    # We must use the REAL x-values (features) to align the plot correctly.
    # Assuming Feature 0 is "Time" or "Location"
    
    xc_np = xc[sample_idx].cpu().detach().numpy() # Context Inputs
    xt_np = xt[sample_idx].cpu().detach().numpy() # Target Inputs
    
    yc_np = yc[sample_idx].cpu().detach().numpy() # Context Truth
    yt_np = yt[sample_idx].cpu().detach().numpy() # Target Truth
    
    mu_np = mu_yt[sample_idx].cpu().detach().numpy() # Prediction Mean
    std_np = torch.sqrt(var_yt[sample_idx] + 1e-8).cpu().detach().numpy() # Uncertainty

    # 2. EXTRACT X-AXIS
    # We take the first feature column as the x-axis. 
    # If your input has multiple features (like lat/lon), this plots against the first one.
    x_context = xc_np[:, 0]
    x_target = xt_np[:, 0]

    # 3. CRITICAL: SORT THE TARGETS
    # The target points are often random. If we plot a line through random points 
    # (e.g., x=10, then x=2), the line will zig-zag. We must sort by X.
    
    sort_idxs = np.argsort(x_target)
    
    # Apply sorting to everything related to Target
    x_target_sorted = x_target[sort_idxs]
    yt_np_sorted = yt_np[sort_idxs]
    mu_np_sorted = mu_np[sort_idxs]
    std_np_sorted = std_np[sort_idxs]

    number_of_output_features = yc_np.shape[-1]

    for dim_idx in range(number_of_output_features):
        var_name = output_cols[dim_idx]
        plt.figure(figsize=(10, 6))

        # --- Plot 1: Context Points (Scatter) ---
        # Context doesn't need sorting because it's just dots (scatter)
        plt.scatter(
            x_context,
            yc_np[:, dim_idx],
            color="blue",
            label="Context (Input)",
            s=50,
            alpha=0.6,
            zorder=5
        )

        # --- Plot 2: Target Truth (Sorted Line + Scatter) ---
        # We plot a line AND small dots so you can see where the actual points are
        plt.plot(
            x_target_sorted,
            yt_np_sorted[:, dim_idx],
            color="red",
            label="Target (Truth)",
            linewidth=2,
            alpha=0.5,
            zorder=2
        )
        plt.scatter(
            x_target_sorted, 
            yt_np_sorted[:, dim_idx], 
            color="red", 
            s=15, 
            alpha=0.4
        )

        # --- Plot 3: Prediction Mean (Sorted Line) ---
        plt.plot(
            x_target_sorted,
            mu_np_sorted[:, dim_idx],
            color="green",
            label="Prediction (Mean)",
            linewidth=2,
            zorder=4
        )

        # --- Plot 4: Uncertainty (Sorted Fill) ---
        lower = mu_np_sorted[:, dim_idx] - 1.96 * std_np_sorted[:, dim_idx]
        upper = mu_np_sorted[:, dim_idx] + 1.96 * std_np_sorted[:, dim_idx]
        
        plt.fill_between(
            x_target_sorted,
            lower,
            upper,
            color="green",
            alpha=0.2,
            label="Uncertainty (95% CI)",
            zorder=3
        )

        plt.title(f"{experiment_name}\nVariable: {var_name}")
        plt.xlabel("Input Feature (e.g., Time/Location)")
        plt.ylabel("Normalized Value")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        file_name = f"{experiment_name}_var_{var_name}.png"
        save_path = Path(save_dir) / file_name
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close()
        
"Making the plots for R2 values "


# def plot_measured_vs_estimated(real_t, real_p, var_name, exp_name, plot_dir):
#     """
#     Scatter/hexbin plot: Measured vs Estimated with 1:1 line, fitted line,
#     and statistics (slope, intercept, R2, RMSE).

#     real_t, real_p: 1D or 2D NumPy arrays (already de-normalized).
#     """
#     # Flatten
#     y_true = np.asarray(real_t).flatten()
#     y_pred = np.asarray(real_p).flatten()

#     # Remove NaNs/inf
#     mask = np.isfinite(y_true) & np.isfinite(y_pred)
#     y_true = y_true[mask]
#     y_pred = y_pred[mask]

#     if len(y_true) == 0:
#         print(f"[WARN] No finite samples for {exp_name} – {var_name}, skipping scatter.")
#         return

#     # Stats
#     r2 = r2_score(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))

#     # Linear fit y = a x + b
#     if len(y_true) >= 2:
#         a, b = np.polyfit(y_true, y_pred, 1)
#         x_line = np.linspace(y_true.min(), y_true.max(), 200)
#         y_line = a * x_line + b
#     else:
#         a, b = 1.0, 0.0
#         x_line = np.array([y_true.min(), y_true.max()])
#         y_line = x_line

#     # For 1:1 line
#     min_val = min(y_true.min(), y_pred.min())
#     max_val = max(y_true.max(), y_pred.max())

#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Option 1: scatter (lighter but good enough)
#     # ax.scatter(y_true, y_pred, s=5, c='blue', alpha=0.3)

#     # Option 2: hexbin for dense data (closer to your paper figure)
#     hb = ax.hexbin(
#         y_true,
#         y_pred,
#         gridsize=80,
#         cmap='turbo',
#         mincnt=1,
#         alpha=0.9
#     )
#     cb = fig.colorbar(hb, ax=ax)
#     cb.set_label("Counts")

#     # 1:1 line (dashed)
#     ax.plot([min_val, max_val], [min_val, max_val],
#             'k--', linewidth=1.0, label="1:1")

#     # Fitted line (solid)
#     ax.plot(x_line, y_line,
#             color='red', linewidth=1.2, label="Fitted")

#     ax.set_xlabel(f"Measured {var_name}")
#     ax.set_ylabel(f"Estimated {var_name}")
#     ax.set_title(f"{exp_name} – {var_name}")

#     # Text box with stats
#     text_str = (
#         f"Y = {a:.2f}X + {b:.2f}\n"
#         f"$R^2$ = {r2:.2f}\n"
#         f"RMSE = {rmse:.2f}\n"
#         f"N = {len(y_true)}"
#     )
#     ax.text(
#         0.05,
#         0.95,
#         text_str,
#         transform=ax.transAxes,
#         fontsize=9,
#         va="top",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
#     )

#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="lower right", fontsize=8)

#     plt.tight_layout()

#     # Save
#     var_safe = var_name.replace(" ", "_").replace("/", "_")
#     fname = f"{exp_name}_measured_vs_estimated_{var_safe}.png"
#     save_path = plot_dir / fname
#     plt.savefig(save_path, dpi=300)
#     print(f"Saved measured-vs-estimated plot: {save_path}")
#     plt.close(fig)
        
        

class NP_Evaluator:
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    # TO evaluate the model we will first upload the best weights.

    def load_best_checkpoints(self, checkpoint_dir_path):
        """Load the best weights from a specific file path."""
        # Here the base name will give us the check points file path last name
        
        print(
            f" >>>>> Loading the best Weights: {os.path.basename(checkpoint_dir_path)}"
        )
        # Here first we will see the maximum time stamp then we will go inside check the best weights and then we will load them. 
        
        list_of_all_the_time_stamps = glob.glob((os.path.join(checkpoint_dir_path, "*")))
        
        # Now we will search for the max time stamp dir
        max_time_stamp_dir = max(list_of_all_the_time_stamps, key = os.path.getmtime)
        
        # Now once we have find the max time stamp directory we will find the best weights 
        
        list_of_weights = glob.glob(os.path.join(max_time_stamp_dir, "*.pth" ))
        
        best_weights = max(list_of_weights, key = lambda f: int(f.split("_")[-1].split(".")[0]) )
        
        checkpoints = torch.load(best_weights, map_location = self.device)
        
        # here we will check if the new instance is in the dictionary format.
        if isinstance(checkpoints, dict) and 'model_state_dict' in checkpoints:
            self.model.load_state_dict(checkpoints['model_state_dict'])
            print(f"Successfully loaded model weights")
        else:
            self.model.load_state_dict(checkpoints, map_location = self.device)



    def calculate_negative_log_liklihood(self, y_t, mu_y_pred, var_y_pred):
        """
        Liklihood = (1/sqrt(2*pi*var) * e^(y_t - mu_y_pred)^2/2 * var_y_pred^2)
        after taking log both side

        NLL = 0.5 * log(vary_pred) + (y_t - mu_y)^2/var_y_pred + 0.5*log(2*pi)

        """

        # Now once we found the file of the best weights we will load them.

        NLL = 0.5 * (
            (torch.log(2 * torch.pi * var_y_pred))
            + ((y_t - mu_y_pred) ** 2 / var_y_pred)
        )
        return NLL

    def run_eval(self, dataloader, output_cols, plot_dir, exp_name, plots_to_make=2):
        # Here we are considering to make 2 plots only as we will plot the graph for 2 batches only.

        all_true = []
        all_mu = []
        all_nll = []
        all_coverage = [] # Here all coverage will be stored as we need tp see the uncertainty
        plots_done = 0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(dataloader, desc="Eval"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                xc, yc, xt, yt = context_target_split(
                    x_batch, y_batch, min_context=5, max_context=15, num_target=25
                )

                mu_yt, var_yt= self.model.predict(xc, yc, xt)
                standerd_deviation = torch.sqrt(var_yt + 1e-8)
                
                # Lower and upper bound to capture the uncertainty in validatiion dataset 
                lower = mu_yt -1.96 * standerd_deviation
                upper = mu_yt + 1.96 * standerd_deviation
                
                # checking how many points falls in this uncertainty level.
                coverage_batch = ((yt>=lower) & (yt <=upper)).float().mean().item()
                all_coverage.append(coverage_batch)
                
                nll_batch_wise = self.calculate_negative_log_liklihood(
                    yt, mu_yt, var_yt
                )
                nll_score_elementwise = nll_batch_wise.mean(dim=1)

                if plots_done < plots_to_make:
                    visualize_sample(
                        xc,
                        yc,
                        xt,
                        yt,
                        mu_yt,
                        var_yt,
                        output_cols,
                        exp_name,
                        0,
                        plot_dir,
                    )
                    plots_done += 1

                all_mu.append(mu_yt.cpu().numpy())
                all_true.append(yt.cpu().numpy())
                all_nll.append(nll_score_elementwise.cpu().numpy())
        return (
            np.concatenate(all_true, axis=0),
            np.concatenate(all_mu, axis=0),
            np.concatenate(all_nll, axis=0),
            np.mean(all_coverage, axis=0) # check this again
        )
