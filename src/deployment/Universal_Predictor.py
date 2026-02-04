# Importing the necessary libraries.
import sys
import os
import torch
import glob
import numpy as np
from pathlib import Path

# Settign the root directory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
print(ROOT)

from src.Models.neural_process import NeuralProcess


class Universal_Predictor:
    def __intit__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Our best model is as per the metric we have provided in the validation data set.
        self.best_experiment_name = "latitude_longitude_AOD_PM2.5____AOD_PM2.5"

        # we have defined the input and output dimensions.

        self.x_dim = 4
        self.y_dim = 2

        self.model = None
        self._load_best_model()

    def _find_best_checkpoint(self):
        best_experiment_dir = ROOT / "cpu_checkpoints" / self.best_experiment_name

        if not os.path.exists(best_experiment_dir):
            raise FileNotFoundError(
                f"Best experiment directory not found: {best_experiment_dir}"
            )
        # Here we have reached to the best experiment directory now we will look in to the latest time.
        best_time_stamp_folders = glob.glob(os.path.join(best_experiment_dir, "*"))
        if not best_time_stamp_folders:
            raise FileNotFoundError(
                f"No timestamp folders found in {best_experiment_dir}"
            )
        best_time_stamp_folders = max()
        latest_time_stamp_folder = max(best_time_stamp_folders, key=os.path.getctime)

        if not best_time_stamp_folders:
            raise FileNotFoundError(
                f"No timestamp folders found in {best_time_stamp_folders}"
            )
        best_weights = glob.glob(os.path.join(latest_time_stamp_folder, "*.pth"))

        if not best_weights:
            raise FileNotFoundError(
                f"No best weight file found in {latest_time_stamp_folder}"
            )
        # Now we will return the

        best_checkpoint = max(
            best_weights, key=lambda f: int(f.split("_")[-1].split(".")[0])
        )

        return best_checkpoint

    # Now once we have found the best_experiment directory we will look for the best weights.

    def _load_best_model(self):
        try:
            best_checkpoint_path = self._final_find_best_checkpoint()

            # Once we are inside the latest time stamp folder we will find the best model weights.
            self.model = NeuralProcess(
                x_dim=self.x_dim, y_dim=self.y_dim, hidden_dim=128, latent_dim=128
            )

            # this is our standerd model now we will load the weights in the model.
            self.model.load_state_dict(
                torch.load(best_checkpoint_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error in loading the best model: {e}")

    # After loading the best model we will predict the values using this model.
    def predict(self, lat: float, long: float, aod: float = None, Pm25: float = None):
        if self.model is None:
            raise ValueError("Model is not loaded properly.")

        # As in our choosen model the input dim size is 4 hence we will provide the vecotr of input size 4.

        # Input vector
        # Initializing the input vecotor
        input_vector = [0.0, 0.0, 0.0, 0.0]
        input_vector[0] = lat
        input_vector[1] = long
        input_vector[2] = aod if aod is not None else 0.0
        input_vector[3] = Pm25 if Pm25 is not None else 0.0
        # Initializing the output vector
        # Output vector
        output_vector = [0.0, 0.0]
        output_vector[0] = aod if aod is not None else 0.0
        output_vector[1] = Pm25 if Pm25 is not None else 0.0

        xc = (
            torch.tensor([input_vector], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        yc = (
            torch.tensor([output_vector], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        xt = (
            torch.tensor([input_vector], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            mu_yt, var_yt = self.model.predict(xc, yc, xt)

        return {
            "Used_model": self.best_experiment_name,
            "input_processed": input_vector,
            "predicted_mean": mu_yt[0, 0, 0].item(),
            "predicted_variance": var_yt[0, 0, 0].item(),
        }


# Max check point file
# import torchvision
# mu = torch.tensor([[[1.0, 2.0]]])
# mu[0, 0, 0]
