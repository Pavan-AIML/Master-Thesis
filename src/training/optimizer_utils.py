# Here we will create the dat aloader and train test split and trained and evaluator.
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import numpy as np
import time
import glob
from sklearn.metrics import r2_score
from tqdm import tqdm

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def context_target_split(x, y, min_context=50, max_context=100, num_target=200):
    x = x.clone().detach()
    y = y.clone().detach()
    batch_size, num_points, _ = x.shape

    # --- FIX STARTS HERE ---
    # 1. Ensure min_context isn't larger than the total points available
    actual_min = min(min_context, num_points - 1)
    
    # 2. Ensure real_max is at least actual_min + 1 so randint doesn't crash
    real_max = min(num_points, max_context)
    
    if actual_min >= real_max:
        num_context = actual_min
    else:
        num_context = int(np.random.randint(actual_min, real_max+1))

    # --- FIX ENDS HERE ---

    indices = torch.randperm(num_points)

    x_context = x[:, indices[:num_context], :]
    y_context = y[:, indices[:num_context], :]

    effective_target_num = min(num_points, num_target)
    target_indices = indices[:effective_target_num]

    x_target = x[:, target_indices, :]
    y_target = y[:, target_indices, :]

    return x_context, y_context, x_target, y_target

"""
From this class we will try to get the final data in the form of data_loader this will take the data form finallloader and then convert it to the dataloader compatible data set

"""


class neural_process_data(Dataset):
    # here the data set will be in tuple form.
    def __init__(self, X_full, Y_full, num_points_per_task=25):
        self.X_full = X_full.detach().cpu()
        self.Y_full = Y_full.detach().cpu()
        self.N = X_full.shape[0]
        self.num_points_per_task = num_points_per_task

    def __len__(self):
        return len(self.X_full)

    def __getitem__(self, index):
        # I will randomly generate the N numbers and then select the numbers of length num_points_per_task
        npts = min(self.N, self.num_points_per_task)
        idxs = torch.randperm(self.N)[: npts]
        X = self.X_full[idxs].clone().detach()
        Y = self.Y_full[idxs].clone().detach()
        return X, Y



"New Training class "




class NPTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,               # e.g. NPELBO(beta=1.0)
        device,
        log_dir="./logs",
        checkpoint_dir="./checkpoints/",
        input_dim=1,
        output_dim=1,
        context_min=50,
        context_max=100,
        num_target=200,
        target_mean=None,      # per-output-dim mean (for de-normalization)
        target_std=None,       # per-output-dim std  (for de-normalization)
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # TensorBoard writer and checkpoint dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.checkpoint_dir = checkpoint_dir

        # NP hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_min = context_min
        self.context_max = context_max
        self.num_target = num_target

        # Store de-normalization stats on device (for RMSE in original units)
        if target_mean is not None and target_std is not None:
            if torch.is_tensor(target_mean):
                self.target_mean = target_mean.to(device).float()
                self.target_std = target_std.to(device).float()
            else:
                self.target_mean = torch.tensor(target_mean, device=device).float()
                self.target_std = torch.tensor(target_std, device=device).float()
        else:
            self.target_mean = None
            self.target_std = None

        # Hyperparameters summary (for hparams logging)
        self.hparams = {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer": type(optimizer).__name__,
            "context_min": context_min,
            "context_max": context_max,
            "num_target": num_target,
            "device": str(device),
            "model_name": type(model).__name__,
        }

        self._log_model_graph()

    def _log_model_graph(self):
        """Log model graph once to TensorBoard (safe to skip on failure)."""
        try:
            dummy_xc = torch.randn(1, self.context_max, self.input_dim, device=self.device)
            dummy_yc = torch.randn(1, self.context_max, self.output_dim, device=self.device)
            dummy_xt = torch.randn(1, self.num_target, self.input_dim, device=self.device)
            dummy_yt = torch.randn(1, self.num_target, self.output_dim, device=self.device)
            self.writer.add_graph(self.model, (dummy_xc, dummy_yc, dummy_xt, dummy_yt))
            print("Successfully logged model graph.")
        except Exception as e:
            print(f"Logging of graph skipped: {e}")

    def train_epoch(self, dataloader, epoch_idx, target_col_names=None):
        self.model.train()

        # Accumulators for metrics
        metric = {
            "total_loss": 0.0,
            "nll": 0.0,
            "kl": 0.0,
            "rmse": 0.0,
            "coverage": 0.0,
            "grad_norm": 0.0,
            "logvar_y": 0.0,
        }
        all_real_t = []
        all_real_p = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}")

        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Split into context and target
            xc, yc, xt, yt = context_target_split(
                x_batch, y_batch,
                min_context=self.context_min,
                max_context=self.context_max,
                num_target=self.num_target,
            )

            self.optimizer.zero_grad()

            # Forward pass: model returns dict with distributions + params
            outputs = self.model(xc, yc, xt, yt)
            mu_y = outputs["mu_y"]        # [B, N_t, output_dim]
            sigma_y = outputs["sigma_y"]  # [B, N_t, output_dim]
            var_y = sigma_y ** 2          # only for metrics/coverage

            # Loss (DeepMind-style ELBO via distributions)
            loss, nll, kl = self.loss_fn(outputs, yt)

            # Backprop + optimization
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            # ---- Metrics (RMSE in original or normalized scale) ----
            with torch.no_grad():
                if self.target_mean is not None and self.target_std is not None:
                    # de-normalize to original units
                    t_mean = self.target_mean.to(self.device)
                    t_std = self.target_std.to(self.device)
                    real_mean = mu_y * t_std + t_mean       # pred in original scale
                    target_y = yt * t_std + t_mean          # true in original scale
                else:
                    real_mean = mu_y
                    target_y = yt

                mse_batch = torch.mean((real_mean - target_y) ** 2)
                rmse_batch = torch.sqrt(mse_batch).item()
                metric["rmse"] += rmse_batch

                all_real_t.append(target_y.detach().cpu().numpy())
                all_real_p.append(real_mean.detach().cpu().numpy())

            # ---- Additional metrics (coverage, logvar_y) ----
            var_safe = torch.clamp(var_y, min=1e-6)
            std_y = torch.sqrt(var_safe)
            lower = mu_y - 1.96 * std_y
            upper = mu_y + 1.96 * std_y
            coverage_batch = ((yt >= lower) & (yt <= upper)).float().mean().item()
            metric["coverage"] += coverage_batch
            metric["logvar_y"] += torch.mean(torch.log(var_safe)).item()

            metric["total_loss"] += loss.item()
            metric["nll"] += nll.item()
            metric["kl"] += kl.item()
            metric["grad_norm"] += grad_norm.item()

            pbar.set_postfix({
                "Loss": f"{metric['total_loss'] / (batch_idx + 1):.2f}",
                "RMSE": f"{metric['rmse'] / (batch_idx + 1):.4f}",
            })

        # --- Aggregate over batches ---
        n_batches = len(dataloader)
        if n_batches > 0:
            for k in metric:
                metric[k] /= n_batches

            # Compute R2 on de-normalized predictions (if available)
            try:
                all_real_t_arr = np.concatenate(all_real_t, axis=0)
                all_real_p_arr = np.concatenate(all_real_p, axis=0)
                r2_value = r2_score(all_real_t_arr.flatten(), all_real_p_arr.flatten())
            except Exception:
                r2_value = float("nan")

            # Log to TensorBoard
            self.writer.add_scalar("Loss/Total", metric["total_loss"], epoch_idx)
            self.writer.add_scalar("Loss/NLL", metric["nll"], epoch_idx)
            self.writer.add_scalar("Loss/KL", metric["kl"], epoch_idx)
            self.writer.add_scalar("Metrics/RMSE", metric["rmse"], epoch_idx)
            self.writer.add_scalar("Metric/Coverage_95", metric["coverage"], epoch_idx)
            self.writer.add_scalar("Grad/Norm", metric["grad_norm"], epoch_idx)
            self.writer.add_scalar("Uncertainty/Mean_LogVar", metric["logvar_y"], epoch_idx)
            self.writer.add_scalar("Metric/R2", r2_value, epoch_idx)

            if epoch_idx % 5 == 0:
                # Use last batch’s xc, yc, xt, yt, mu_y, var_y for a diagnostic plot
                self.plot_prediction_to_tensorboard(
                    xc, yc, xt, yt, mu_y, var_y, epoch_idx, target_col_names=target_col_names
                )

            self.writer.flush()

        return metric["total_loss"]

    def evaluate(self, val_dataloader):
        """Compute average validation loss over a dataloader."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                xc, yc, xt, yt = context_target_split(
                    x_batch, y_batch,
                    min_context=self.context_min,
                    max_context=self.context_max,
                    num_target=self.num_target,
                )

                outputs = self.model(xc, yc, xt, yt)
                loss, _, _ = self.loss_fn(outputs, yt)
                running_loss += loss.item()
                n_batches += 1

        avg_val_loss = running_loss / max(1, n_batches)
        self.model.train()
        return avg_val_loss

    def plot_prediction_to_tensorboard(self, xc, yc, xt, yt, mu, var, epoch, target_col_names=None):
        """Plot predictions+uncertainty for one batch and log to TensorBoard."""


        # Take first task in batch
        b_xc = xc[0].detach().cpu().numpy()
        b_yc = yc[0].detach().cpu().numpy()
        b_xt = xt[0].detach().cpu().numpy()
        b_yt = yt[0].detach().cpu().numpy()
        b_mu = mu[0].detach().cpu().numpy()
        b_std = np.sqrt(var[0].detach().cpu().numpy())

        output_dim = b_yt.shape[-1]
        fig, axes = plt.subplots(output_dim, 1, figsize=(12, 6 * output_dim))
        if output_dim == 1:
            axes = [axes]

        x_range = np.arange(b_yt.shape[0])  # index per target point

        for i in range(output_dim):
            ax = axes[i]
            col_name = target_col_names[i] if target_col_names else f"Feature_{i}"

            ax.plot(x_range, b_yt[:, i], "k--", alpha=0.6, label="Ground Truth")
            ax.plot(x_range, b_mu[:, i], color="blue", linewidth=2, label="Prediction")

            ax.fill_between(
                x_range,
                b_mu[:, i] - 1.96 * b_std[:, i],
                b_mu[:, i] + 1.96 * b_std[:, i],
                color="blue",
                alpha=0.15,
                label="95% CI",
            )

            # Mark context points
            for j in range(len(b_xc)):
                idx = np.where((b_xt == b_xc[j]).all(axis=1))[0]
                if len(idx) > 0:
                    ax.scatter(
                        idx[0],
                        b_yc[j, i],
                        color="black",
                        marker="o",
                        s=50,
                        edgecolors="white",
                        zorder=5,
                        label="Context Point" if j == 0 else "",
                    )

            ax.set_title(f"{col_name} - Prediction")
            ax.set_xlabel("Target index")
            ax.set_ylabel("Value (normalized or original)")
            ax.legend(loc="upper right", fontsize="small", ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure("Visuals/Predictions", fig, global_step=epoch)
        plt.close(fig)
        
    def plot_scatter_true_vs_pred(self, y_true, y_pred, epoch, var_name="Variable"):
        """
        Scatter plot: true vs predicted with fitted line and R^2, like in scientific papers.
        y_true, y_pred: 1D NumPy arrays or tensors with same length.
        var_name: label for axis/title (e.g. 'PM2.5 (µg/m³)' or 'AOD').
        """
        # Convert to 1D NumPy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        # Remove NaNs if any
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        # Compute R^2
        r2 = r2_score(y_true, y_pred)
        # Simple linear fit (y_pred = a * y_true + b)
        if len(y_true) >= 2:
            a, b = np.polyfit(y_true, y_pred, 1)
            x_line = np.linspace(y_true.min(), y_true.max(), 100)
            y_line = a * x_line + b
        else:
            a, b = 1.0, 0.0
            x_line = np.array([y_true.min(), y_true.max()])
            y_line = x_line
        # Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_true, y_pred, s=15, color="black", alpha=0.7, label="Samples")
        # 1:1 line (perfect agreement)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, label="1:1 line")
        # Fitted regression line
        ax.plot(x_line, y_line, color="red", linewidth=1.5, label="Fitted line")
        ax.set_xlabel(f"Observed {var_name}")
        ax.set_ylabel(f"Predicted {var_name}")
        ax.set_title(f"{var_name}: Observed vs Predicted")
        # Text box with R^2 and N
        text_str = f"$R^2 = {r2:.3f}$\nN = {len(y_true)}"
        ax.text(
            0.05,
            0.95,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        # Log to TensorBoard
        self.writer.add_figure(f"Scatter/{var_name}_true_vs_pred", fig, global_step=epoch)
        plt.close(fig)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hparams": self.hparams,
            },
            path,
        )
        print(f"Checkpoint saved at: {path}")

    def log_final_hparams(self, final_metric_value):
        self.writer.add_hparams(self.hparams, {"Metrics/RMSE": final_metric_value})
        self.writer.close()
        print(f"Final hparams logged with RMSE: {final_metric_value}")


