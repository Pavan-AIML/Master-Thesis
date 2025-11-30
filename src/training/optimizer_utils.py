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


class NeuralProcessDataset(Dataset):
    """
    Creates context and target sets on the fly for Neural Processes.
    X: [N, x_dim]
    Y: [N, y_dim]
    """

    def __init__(
        self,
        X,
        Y,
        min_context=10,
        max_context=50,
        min_target=50,
        max_target=100,
        # we will take 3000 points in all the epochs randomly.
        tasks_per_epoch=3000,
    ):
        self.X = X  # [N, x_dim]
        self.Y = Y  # [N, y_dim]
        self.N = X.shape[0]

        self.min_context = min_context
        self.max_context = max_context
        self.min_target = min_target
        self.max_target = max_target
        self.tasks_per_epoch = tasks_per_epoch

    def __len__(self):
        # Number of NP "tasks" per epoch
        return self.tasks_per_epoch

    def __getitem__(self, idx):
        # Random permutation of indices
        perm = torch.randperm(self.N)

        # Sample random sizes
        C = torch.randint(self.min_context, self.max_context + 1, (1,)).item()
        T = torch.randint(self.min_target, self.max_target + 1, (1,)).item()

        # Select context
        ctx_idx = perm[:C]
        x_c = self.X[ctx_idx]  # [C, x_dim]
        y_c = self.Y[ctx_idx]  # [C, y_dim]

        # Select target
        tgt_idx = perm[C : C + T]
        x_t = self.X[tgt_idx]  # [T, x_dim]
        y_t = self.Y[tgt_idx]  # [T, y_dim]

        return x_c, y_c, x_t, y_t


"""
From this class we will create the class for training the model
"""


class NeuralProcessTrainer:
    def __init__(
        self,
        raw_dataset,
        model_class,
        loss_class,
        x_dim,
        y_dim,
        hidden_dim=128,
        latent_dim=128,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32,
        tasks_per_epoch=3000,
        run_name="NP_Run",
        log_dir="runs",
        ckpt_dir="checkpoints",
        lr=1e-3,
        beta=1.0,
    ):
        """
        raw_dataset: e.g. final_data_latlon_AOD_PM25 = [X, Y]
                     X: [N, x_dim], Y: [N, y_dim]
        model_class: e.g. NeuralProcess
        loss_class : e.g. LossFunctions
        """

        self.X = raw_dataset[0]  # [N, x_dim]
        self.Y = raw_dataset[1]  # [N, y_dim]
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.tasks_per_epoch = tasks_per_epoch
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.lr = lr
        self.beta = beta
        self.run_name = run_name
        self.ckpt_dir = ckpt_dir

        # ---------------- Device ----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Split raw data: train / val / test ----------------
        N = self.X.shape[0]

        # train+val vs test
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            self.X, self.Y, test_size=(1 - train_ratio), random_state=42
        )

        # train vs val inside train_val
        val_ratio_adj = val_ratio / train_ratio
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=val_ratio_adj, random_state=42
        )

        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test

        # ---------------- NP datasets ----------------
        self.train_dataset = NeuralProcessDataset(
            self.X_train,
            self.Y_train,
            tasks_per_epoch=tasks_per_epoch,
        )
        self.val_dataset = NeuralProcessDataset(
            self.X_val,
            self.Y_val,
            tasks_per_epoch=tasks_per_epoch // 2,
        )
        self.test_dataset = NeuralProcessDataset(
            self.X_test,
            self.Y_test,
            tasks_per_epoch=tasks_per_epoch // 2,
        )

        # ---------------- Data loaders ----------------
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

        # ---------------- Model, loss, optim ----------------
        self.model = model_class(
            x_c_dim=self.x_dim,
            y_c_dim=self.y_dim,
            x_t_dim=self.x_dim,
            y_t_dim=self.y_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        self.loss_fn = loss_class(beta=self.beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )

        # ---------------- TensorBoard writer ----------------
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.tb_logdir = os.path.join(log_dir, f"{run_name}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_logdir)

        # ---------------- Checkpoint dir ----------------
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_val_loss = float("inf")

        print(f"TensorBoard logs at: {self.tb_logdir}")
        print(f"Checkpoints will be saved to: {self.ckpt_dir}")

    # ------------- One training epoch -------------

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0

        for step, (x_c, y_c, x_t, y_t) in enumerate(self.train_loader):
            x_c = x_c.to(self.device)
            y_c = y_c.to(self.device)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)

            mu_y, var_y, mu_c, log_var_c, mu_ct, log_var_ct = self.model(
                x_c, y_c, x_t, y_t
            )

            total_loss_batch, nll, kl = self.loss_fn(
                mu_y, var_y, y_t, mu_c, log_var_c, mu_ct, log_var_ct
            )

            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_nll += nll.item()
            total_kl += kl.item()

            global_step = epoch * len(self.train_loader) + step
            # Batch-level logging
            self.writer.add_scalar(
                "Batch/TotalLoss", total_loss_batch.item(), global_step
            )
            self.writer.add_scalar("Batch/NLL", nll.item(), global_step)
            self.writer.add_scalar("Batch/KL", kl.item(), global_step)
            self.writer.add_scalar(
                "Batch/LR", self.optimizer.param_groups[0]["lr"], global_step
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_nll = total_nll / len(self.train_loader)
        avg_kl = total_kl / len(self.train_loader)

        # Epoch-level logging
        self.writer.add_scalar("Epoch/Train_TotalLoss", avg_loss, epoch)
        self.writer.add_scalar("Epoch/Train_NLL", avg_nll, epoch)
        self.writer.add_scalar("Epoch/Train_KL", avg_kl, epoch)

        return avg_loss, avg_nll, avg_kl

    # ------------- Evaluation (val/test) -------------

    def evaluate(self, loader, epoch=None, split_name="Val"):
        self.model.eval()
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0

        with torch.no_grad():
            for x_c, y_c, x_t, y_t in loader:
                x_c = x_c.to(self.device)
                y_c = y_c.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)

                mu_y, var_y, mu_c, log_var_c, mu_ct, log_var_ct = self.model(
                    x_c, y_c, x_t, y_t
                )

                total_loss_batch, nll, kl = self.loss_fn(
                    mu_y, var_y, y_t, mu_c, log_var_c, mu_ct, log_var_ct
                )

                total_loss += total_loss_batch.item()
                total_nll += nll.item()
                total_kl += kl.item()

        avg_loss = total_loss / len(loader)
        avg_nll = total_nll / len(loader)
        avg_kl = total_kl / len(loader)

        if epoch is not None:
            self.writer.add_scalar(f"Epoch/{split_name}_TotalLoss", avg_loss, epoch)
            self.writer.add_scalar(f"Epoch/{split_name}_NLL", avg_nll, epoch)
            self.writer.add_scalar(f"Epoch/{split_name}_KL", avg_kl, epoch)

        return avg_loss, avg_nll, avg_kl

    # ------------- Checkpoint saving -------------

    def save_checkpoint(self, epoch, is_best=False):
        ckpt_path = os.path.join(
            self.ckpt_dir,
            f"{self.run_name}_epoch{epoch + 1}{'_best' if is_best else ''}.pth",
        )
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # ------------- Prediction + uncertainty (using context only) -------------

    def predict_with_uncertainty(self, x_c, y_c, x_t, num_samples=20):
        """
        Uses q(z|C) (prior conditioned on context only) and samples multiple z
        to estimate predictive mean and variance at target points.
        x_c: [C, x_dim], y_c: [C, y_dim]
        x_t: [T, x_dim]
        Returns: pred_mean [T, y_dim], pred_var [T, y_dim]
        """
        self.model.eval()
        x_c = x_c.to(self.device).unsqueeze(0)  # [1, C, x_dim]
        y_c = y_c.to(self.device).unsqueeze(0)  # [1, C, y_dim]
        x_t = x_t.to(self.device).unsqueeze(0)  # [1, T, x_dim]

        with torch.no_grad():
            # Use encoder + latent_encoder manually for prior q(z|C)
            rc = self.model.encoder(x_c, y_c)  # [1, hidden_dim]
            mu_zc, log_var_zc = self.model.latent_encoder(rc)
            std_zc = torch.exp(0.5 * log_var_zc)  # [1, latent_dim]

            preds = []

            for _ in range(num_samples):
                eps = torch.randn_like(std_zc)
                z = mu_zc + eps * std_zc  # [1, latent_dim]
                mu_y, _ = self.model.decoder(x_t, z)  # [1, T, y_dim]
                preds.append(mu_y)  # each: [1, T, y_dim]

            preds = torch.stack(preds, dim=0)  # [S, 1, T, y_dim]
            preds = preds.squeeze(1)  # [S, T, y_dim]

            pred_mean = preds.mean(dim=0)  # [T, y_dim]
            pred_var = preds.var(dim=0)  # [T, y_dim]

        return pred_mean.cpu(), pred_var.cpu()

    # ------------- Visualization: log prediction to TensorBoard -------------

    def log_prediction_example(self, epoch):
        """
        Takes one batch from val_loader, picks first sample,
        logs true vs predicted (for first output dim) into TensorBoard.
        """
        self.model.eval()
        try:
            x_c, y_c, x_t, y_t = next(iter(self.val_loader))
        except StopIteration:
            return

        x_c = x_c[0]  # [C, x_dim]
        y_c = y_c[0]  # [C, y_dim]
        x_t = x_t[0]  # [T, x_dim]
        y_t = y_t[0]  # [T, y_dim]

        pred_mean, pred_var = self.predict_with_uncertainty(
            x_c, y_c, x_t, num_samples=20
        )

        # Only plot first output dimension
        true_vals = y_t[:, 0].cpu().numpy()
        pred_vals = pred_mean[:, 0].cpu().numpy()
        std_vals = pred_var[:, 0].sqrt().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(true_vals, label="True", marker="o")
        ax.plot(pred_vals, label="Pred mean", marker="x")
        ax.fill_between(
            range(len(pred_vals)),
            pred_vals - 2 * std_vals,
            pred_vals + 2 * std_vals,
            alpha=0.2,
            label="±2 std",
        )
        ax.set_title("Prediction vs True (dim 0)")
        ax.legend()

        self.writer.add_figure("Prediction/example_dim0", fig, global_step=epoch)
        plt.close(fig)

    # ------------- Main training loop -------------

    def train(self, epochs=50, log_pred_every=10):
        for epoch in range(epochs):
            train_loss, train_nll, train_kl = self.train_one_epoch(epoch)
            val_loss, val_nll, val_kl = self.evaluate(
                self.val_loader, epoch=epoch, split_name="Val"
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss={train_loss:.4f} (NLL={train_nll:.4f}, KL={train_kl:.4f}) | "
                f"Val Loss={val_loss:.4f} (NLL={val_nll:.4f}, KL={val_kl:.4f})"
            )

            # Checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # Log prediction example occasionally
            if (epoch + 1) % log_pred_every == 0:
                self.log_prediction_example(epoch)

            # LR scheduler step
            self.scheduler.step()

        # Final test evaluation
        test_loss, test_nll, test_kl = self.evaluate(
            self.test_loader, epoch=None, split_name="Test"
        )
        print(f"Final TEST Loss={test_loss:.4f} (NLL={test_nll:.4f}, KL={test_kl:.4f})")

        self.writer.close()
        return self.model
