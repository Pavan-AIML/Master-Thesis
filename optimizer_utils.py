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
        input_dim = 1,
        output_dim = 1,
        context_min = 20,
        context_max = 50,
        num_target=200,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conetxt_min= context_min
        self.context_max = context_max
        self.num_target = num_target
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.hparams = {
        "learning_rate": optimizer.param_groups[0]['lr'],
        "optimizer":type(optimizer).__name__,
        "context_min":context_min,
        "context_max":context_max,
        "num_target": num_target,
        "device":str(device),
        "model_name":str(model.__class__.__name__)
    }   # This is the dummy metric where we will store all the hyperparameters.
        self.writer.add_hparams(self.hparams, {"Metrics/MSE": 0})
        # Here we also need a dummy metric where we can store the logs
        
        self._log_model_graph()
        
    def _log_model_graph(self):
        " Log model computational graph once"
        try:
            dummy_xc = torch.randn(1,self.context_max, self.input_dim,device = self.device)
            dummy_yc = torch.randn(1, self.context_max, self.output_dim, device = self.device)
            dummy_xt = torch.randn(1, self.num_target, self.input_dim, device = self.device)
            # yt is usually only needed for training /loss, but some NP models use it in forward pass.
            dummy_yt = torch.randn(1, self.num_target, self.output_dim, device = self.device)
            self.writer.add_graph(self.mode, (dummy_xc, dummy_yc, dummy_xt, dummy_yt))
            print("successfully logged model graph")
        except Exception as e:
            print(f"logging of graph is skipped {e}")
            
        
    def train_epoch(self, dataloader, epoch_idx):
        # Putting the mdoel in training state.
        self.model.train()
        
        # Lisiting all the metrices we need.
        
        metric = {
            "total_loss" : 0, "nll":0, "kl":0, "mse":0, 
            "coverage":0, "grad_norm":0, "logvar_y":0
        }
        

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}")

        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            # Move to device
            x_batch = x_batch.clone().detach().to(self.device)
            y_batch = y_batch.clone().detach().to(self.device)
            

            # --- SPLIT DATA ---
            # we can use the functions defined outside of the class also

            xc, yc, xt, yt = context_target_split(
                x_batch, y_batch, min_context=self.conetxt_min
                , max_context=self.context_max, num_target=self.num_target
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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # --- METRICS ---
            metric["total_loss"] += loss.item()
            metric["nll"] += nll.item()
            metric["kl"] += kl.item()
            metric["grad_norm"] += grad_norm.item()
            
            # Metric MSE 
            metric["mse"] += torch.mean((mu_y-yt)**2).item()
            
            # 95 % confidence interval 
            std_y = torch.sqrt(var_y + 1e-8)
            lower , upper = mu_y - 1.96 * std_y, mu_y + 1.96 * std_y
            metric["coverage"] += torch.mean((yt>=lower) & yt<=upper).float().item()
            metric["logvar_y"] += torch.mean(torch.log(var_y + 1e-8)).item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.2f}", "MSE":f"{metric['mse']/(batch_idx+1):.4f}"})

        # --- LOGGING ---
        n_batches = len(dataloader)
        if n_batches > 0:
            for k in metric:metric[k] /=n_batches
            
            # Basic Losses 
            self.write.add_scaler("Loss/Total", metric["total_loss"], epoch_idx)
            self.writer.add_scaler("Loss/NLL", metric["nll"], epoch_idx)
            self.writer.add_scaler("Loss/KL", metric["kl"], epoch_idx)
            
            # Performace and Stability
            
            self.writer.add_scaler("Metrics/MSE", metric["mse"], epoch_idx)
            self.writer.add_scaler("Metric/Coverage_95", metric["coverage"], epoch_idx)
            self.writer.add_scaler("Grad/Norm", metric["grad_norm"], epoch_idx)
            self.writer.add_scaler("Uncertainty/Mean_LogVar", metric["logvar_y"], epoch_idx)
            
            # Latent space diagnostics 
            self.writer.add_scaler("Latent/Mu_zc_man", mu_zc.mean().item(), epoch_idx)
            if epoch_idx % 5 ==0:
                self.write.add_histogram("Latent/zc_dict", mu_zc.detach().cpu(), epoch_idx)
                self.plot_prediction_to_tensorboard(xc[0], yc[0], xt[0], yt[0], mu_y[0], var_y[0], epoch_idx)
        
        return metric["total_loss"]
    

        # Here we are saving the logs by averaging the losses over all the batches for each epoch. AND saving them in the self.writer.add_scaler in the tensor board

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            "epoch":epoch,
            # Here we save the model weights.
            "model_state_dict":self.model.state_dict(),
            # Here we save the training history of the model.
            "optimizer_state_dict":self.optimizer.state_dict(),
            # here we save the model's hyperparameters.
            "hparams":self.hparams
                    }, path)
        print(f"Checkpoints,epoch, optm  saved: {path}")
        


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
