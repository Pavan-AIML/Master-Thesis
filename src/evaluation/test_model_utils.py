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
from training.optimizer_utils import neural_process_data, context_target_split

# Check the functions.


# ROOT = Path(__file__).resolve().parents[2]
# ROOT
# # Evaluator class
# check_point_dir = ROOT / "cpu_checkpoints" / "{model_name}" / "{time_stamp}"
# check_point_dir
# files = glob.glob(os.path.join(check_point_dir, "*.pth"))
# files
# latest_file = max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
# latest_file


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

    # Move to CPU/Numpy for plotting
    yc_np = yc[sample_idx].cpu().detach().numpy()
    yt_np = yt[sample_idx].cpu().detach().numpy()
    mu_np = mu_yt[sample_idx].cpu().detach().numpy()
    std_np = torch.sqrt(var_yt[sample_idx]).cpu().detach().numpy()

    num_outputs = yc_np.shape[-1]
    num_context = yc_np.shape[0]
    num_target = yt_np.shape[0]

    x_idx_context = np.arange(0, num_context)
    x_idx_target = np.arange(num_context, num_context + num_target)

    for dim_idx in range(num_outputs):
        var_name = output_cols[dim_idx]
        plt.figure(figsize=(10, 6))

        # plotting the context points
        plt.scatter(
            x_idx_context,
            yc_np[:, dim_idx],
            color="blue",
            label="Context (Input)",
            s=50,
            alpha=0.6,
        )
        # known target points vsualization.
        plt.scatter(
            x_idx_target,
            yt_np[:, dim_idx],
            color="red",
            marker="x",
            label="Target (Truth)",
            s=60,
        )
        # Predicted points visualization.
        plt.plot(
            x_idx_target,
            mu_np[:, dim_idx],
            color="green",
            label="Prediction (Mean)",
            linewidth=2,
        )
        # # Lower and Upper boutnds for the uncertainty,
        # after 2 standerd deviations.
        lower = mu_np[:, dim_idx] - 2 * std_np[:, dim_idx]
        upper = mu_np[:, dim_idx] + 2 * std_np[:, dim_idx]
        plt.fill_between(
            x_idx_target,
            lower,
            upper,
            color="green",
            alpha=0.2,
            label="Uncertainty (95% CI)",
        )

        plt.title(f"{experiment_name}\nVariable: {var_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = f"{save_dir}/{experiment_name}_var_{var_name}.png"
        plt.savefig(save_path)
        plt.close()


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
        self.model.load_state_dict(
            torch.load(checkpoint_dir_path, map_location=self.device)
        )

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

    def run_eval(self, dataloader, output_cols, exp_name, plots_to_make=2):
        # Here we are considering to make 2 plots only as we will plot the graph for 2 batches only.
        all_true = []
        all_mu = []
        all_nll = []
        plots_done = 0
        plot_dir = ROOT / "final_plots"

        with torch.no_grad():
            for x_batch, y_batch in tqdm(dataloader, desc="Eval"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                xc, yc, xt, yt = context_target_split(
                    x_batch, y_batch, min_context=50, max_context=100, num_target=200
                )

                mu_yt, var_yt = self.model.predict(xc, yc, xt)
                nll_batch_wise = self.calculate_negative_log_liklihood(
                    yt, mu_yt, var_yt
                )
                nll_score_elementwise = nll_batch_wise.mean(dim=[1, 2])

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
        )
