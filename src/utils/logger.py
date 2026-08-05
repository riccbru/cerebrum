import csv
import json
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(
            self,
            config=None,
            exp_name="3dcnn_adni",
            base_results="results",
            base_checkpoints="checkpoints"
        ):
        # 1. Get Job Slurm ID with timestamp
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}_job{slurm_job_id}"
        
        # 2. Create paths for the specified run
        self.checkpoint_dir = os.path.join(base_checkpoints, self.run_id)
        self.results_dir = os.path.join(base_results, self.run_id)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Output files
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.config_path = os.path.join(self.checkpoint_dir, "config.json")
        self.csv_path = os.path.join(self.results_dir, "metrics_log.csv")
        self.summary_path = os.path.join(self.results_dir, "summary.json")
        
        # 3. Save config in JSON file
        self.config = config or {}
        self.save_config()
        
        # 4. Setup CSV file
        self._init_csv()
        
        print("\033[1;95m[~]\033[0m Experiment started")
        print(f"    ├── Results:     {self.results_dir}")
        print(f"    └── Checkpoints: {self.checkpoint_dir}")

    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def _init_csv(self):
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

    def save_best_model(self, model, epoch, val_loss, val_acc):
        import torch
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config
        }
        torch.save(state, self.best_model_path)
        
        # Update summary with best score
        summary = {
            "run_id": self.run_id,
            "best_epoch": epoch,
            "best_val_loss": val_loss,
            "best_val_acc": val_acc
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(
            f"\033[1;92m[*]\033[0m Best model saved in \033[1m{self.best_model_path}\033[0m \033[90m(Val Loss: {val_loss:.4f})\033[0m"
        )