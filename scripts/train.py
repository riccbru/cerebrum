import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

# Adds repo root directory and imports src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import BrainNiiDataset
from src.models import Simple3DCNN
from src.utils import ExperimentLogger, load_config, set_seed


def train():
    print("Starting training...")
    
    # 1. Loading YAML conf & setting seeds
    config_path = "configs/config_adni.yaml"
    if not os.path.exists(config_path):
        config_path = "../configs/config_adni.yaml"
        
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    exp_logger = ExperimentLogger(
        exp_name=config.get("experiment_name", "3dcnn_adni"),
        config=config,
    )

    # Device setup (GPU / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: \033[1m{device}\033[0m")

    # 2. Dataset loading & anti-leakage split by rid
    csv_path = config["data"]["csv_path"]
    full_dataset = BrainNiiDataset(
        csv_file=csv_path,
        spatial_size=tuple(config["data"]["spatial_size"])
    )
    df = full_dataset.df

    # Grouping by 'rid' to avoid same patient is in both Train e Validation sets (data leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config["training"]["seed"])
    train_idx, val_idx = next(gss.split(df, groups=df['rid']))

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    print(f"DATASET:\n\tTOT: {len(full_dataset)}\n\tTRAIN: {len(train_subset)}\n\tVALIDATION: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=config["training"]["num_workers"]
    )

    # 3. Setup model, loss fun with class weights and optimizer with scheduler
    model = Simple3DCNN(num_classes=config["model"]["num_classes"]).to(device)
    
    # --- Weighted CrossEntropy Loss for class imbalance ---
    train_labels = [full_dataset.df.iloc[i]["label"] for i in train_idx]
    class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer & LR Scheduler ---
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training loop
    best_val_loss = float('inf')
    epochs = config["training"]["epochs"]

    print("\033[1;95m[~]\033[0m Starting training epochs...")
    for epoch in range(1, epochs + 1):
        # --- PHASE: TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # --- PHASE: VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Update Learning Rate Scheduler
        scheduler.step(epoch_val_loss)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}]\n"
            f"\tTrain Loss: {epoch_train_loss:.4f}\n"
            f"\tTrain Acc: {epoch_train_acc:.4f}\n"
            f"\tVal Loss: {epoch_val_loss:.4f}\n"
            f"\tVal Acc: {epoch_val_acc:.4f}"
        )

        exp_logger.log_epoch(
            epoch,
            epoch_train_loss,
            epoch_train_acc,
            epoch_val_loss,
            epoch_val_acc,
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            exp_logger.save_checkpoint(
                model, epoch, epoch_val_loss, epoch_val_acc
            )

    print(
        f"\033[1;92m[*]\033[0m Training complete\nAll artifacts saved under folder: {exp_logger.results_dir}"
    )

if __name__ == "__main__":
    train()