# Here we will create the dat aloader and train test split and trained and evaluator.
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import time

# tensor board
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# x_c = torch.tensor([[1, 2, 3], [4, 5, 6]])
# y_c = torch.tensor([[7, 8, 9], [10, 11, 12]])
# x_t = torch.tensor([[13, 14, 15], [14, 15, 16]])
# y_t = torch.tensor([[17, 18, 19], [20, 21, 22]])

"""
From this below class we will split the data sets in to context and targets.
"""


import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm


def context_target_split(x, y, min_context=5, max_context=100, num_target=1000):
    """
    Splits data into Context and Target.
    Limits Target set size to 'num_target' to prevent Memory Explosion on Mac.
    """
    x = x.clone().detach()
    y = y.clone().detach()
    batch_size, num_points, _ = x.shape

    # Safety: ensure we don't ask for more context than points exist
    real_max = min(num_points - 1, max_context)
    num_context = np.random.randint(min_context, real_max)

    # Shuffle indices
    indices = torch.randperm(num_points)

    # Create Context (Input to Encoder)
    x_context = x[:, indices[:num_context], :]
    y_context = y[:, indices[:num_context], :]

    # Create Target (Input to Decoder / Ground Truth for Loss)
    # We take the Context + Random Extra Points (up to num_target total)
    effective_target_num = min(num_points, num_target)
    target_indices = indices[:effective_target_num]

    x_target = x[:, target_indices, :]
    y_target = y[:, target_indices, :]

    return x_context, y_context, x_target, y_target


"""
From this class we will try to get the final data in the form of data_loader this will take the data form finallloader and then convert it to the dataloader compatible data set

"""


class neural_process_data(Dataset):
    # here the data set will be in tupe form.
    def __init__(self, X_full, Y_full, num_points_per_task=200):
        self.X_full = X_full.clone().detach()
        self.Y_full = Y_full.clone().detach()
        self.N = X_full.shape[0]
        self.num_points_per_task = num_points_per_task

    def __len__(self):
        return len(self.X_full)

    def __getitem__(self, index):
        idxs = torch.randperm(self.N)[: self.num_points_per_task]

        X = self.X_full[idxs].clone().detach()
        Y = self.Y_full[idxs].clone().detach()
        return X, Y


"""
From this class we will create the class for training the model
"""
# model inputs self, x_c_dim, y_c_dim, x_t_dim, y_t_dim, hidden_dim, latent_dim


class NPTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        log_dir="./logs",
        checkpoint_dir="./checkpoints/",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss_avg = 0
        nll_avg = 0
        kl_avg = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}")

        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            # Move to device
            x_batch = x_batch.clone().detach()
            y_batch = y_batch.clone().detach()

            # --- SPLIT DATA ---
            # we can use the functions defined outside of the class also

            xc, yc, xt, yt = context_target_split(
                x_batch, y_batch, min_context=20, max_context=50, num_target=200
            )

            self.optimizer.zero_grad()

            # --- FORWARD PASS ---

            # model inputs -->

            # Returns: (mu_y, var_y, mu_zc, log_var_zc, mu_zct, log_var_zct)
            outs = self.model(xc, yc, xt, yt)
            mu_y, var_y, mu_zc, log_var_zc, mu_zct, log_var_zct = outs

            # self, mu_y, var_y, y_target, mu_zc, mu_zct, log_var_zc, log_var_zct
            # --- LOSS CALCULATION ---

            loss, nll, kl = self.loss_fn(
                mu_y, var_y, yt, mu_zc, mu_zct, log_var_zc, log_var_zct
            )

            # --- OPTIMIZATION ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # --- METRICS ---
            total_loss_avg += loss.item()
            nll_avg += nll.item()
            kl_avg += kl.item()

            pbar.set_postfix({"Loss": f"{loss.item():.2f}"})

        # --- LOGGING ---
        n_batches = len(dataloader)
        if n_batches > 0:
            total_loss_avg /= n_batches
            nll_avg /= n_batches
            kl_avg /= n_batches

        # Here we are saving the logs by averaging the losses over all the batches for each epoch. AND saving them in the self.writer.add_scaler in the tensor board.

        self.writer.add_scalar("Loss/Total", total_loss_avg, epoch_idx)
        self.writer.add_scalar("Loss/NLL", nll_avg, epoch_idx)
        self.writer.add_scalar("Loss/KL", kl_avg, epoch_idx)

        return total_loss_avg

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")


# we will create the validation function for the validation data set.

def validation_function(model, val_dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            xc, yc, xt, yt = context_target_split(
                x_batch, y_batch, min_context=5, max_context=100, num_target=1000
            )

            outs = model(xc, yc, xt, yt)
            mu_y, var_y, mu_zc, log_var_zc, mu_zct, log_var_zct = outs

            loss, nll, kl = loss_fn(
                mu_y, var_y, yt, mu_zc, mu_zct, log_var_zc, log_var_zct
            )
            running_loss += loss.item()

        average_val_loss = running_loss / len(val_dataloader)

        model.train()
        return average_val_loss


# here ww will build the class for the best model selectioin in validation dataset.
import glob

# We 
class Trained_model_selection_in_val_data_set:
    def __init__(self, model, val_dataloader, device, Loss):
        # here model will be a class of model wiith the input arguments.
        self.model = model
        # this dataloader will be the data set of the validation set.
        self.dataloader = val_dataloader
        self.device = device
        # self.context_target_split = context_target_split
        self.Loss = Loss  # Loss class

        """
        model : Neural process model 
        val_data_loader : data loader of validation data set
        device : "cpu"
        Loss: Loss calss that has been made LossFunctions without the forward arguments 
        context_target_split : funtion which splits our data in to context and targets.
        
        """

    def get_the_best_model(self, checkpoint_folder_path):
        # first we will get the x and y in batches so that we can split them in to context and test points and then we can use them for calculating the loss and then calculating the best model.
        # folder of weights file.
        All_check_point_loss = []

        weight_files = glob.glob(os.path.join(checkpoint_folder_path, "*.pth"))

        for file_path in weight_files:
            try:
                self.model.load_state_dict(
                    torch.load(file_path, map_location=self.device)
                )
            except Exception as e:
                print(f"error in loading the file {file_path}:{e}")
                continue
            self.model = self.model.eval()
            running_loss = 0

            with torch.no_grad():
                for x_batch, y_batch in self.dataloader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    xc, yc, xt, yt = context_target_split(x_batch, y_batch)

                    # here we will load the model weights

                    # forward pass
                    outs = self.model(xc, yc, xt, yt)
                    mu_y, var_y, mu_zc, log_var_zc, mu_zct, log_var_zct = outs

                    # loss in the test set
                    # loss forward function.

                    total_loss, nll, kl = self.Loss(
                        mu_y, var_y, yt, mu_zc, mu_zct, log_var_zc, log_var_zct
                    )
                    # here to accumulate the scaler loss and that is the reason we use loss.item()
                    running_loss += total_loss.item()
            # total average loss for each weight checkpoint

            total_avg_loss = running_loss / len(self.dataloader)
            print(f"Total_loss for the {file_path} is : {total_avg_loss}")
            All_check_point_loss.append((total_avg_loss, file_path))
            All_check_point_loss.sort(key=lambda x: x[0])

            best_loss, best_model = All_check_point_loss[0]
        return best_model


