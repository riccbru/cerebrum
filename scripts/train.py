import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

# Aggiunge la radice del repository al path di Python per importare da src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config, set_seed
from src.dataset import BrainNiiDataset
from src.models import Simple3DCNN

def train():
    print("🚀 Avvio della pipeline di addestramento 3D CNN...")
    
    # 1. Caricamento configurazione YAML e fissaggio dei Seed
    config_path = "configs/config_adni.yaml"
    if not os.path.exists(config_path):
        config_path = "../configs/config_adni.yaml"
        
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    # Setup del device (GPU / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: \033[1m{device}\033[0m")

    # 2. Caricamento Dataset e Split anti-leakage per soggetto (rid)
    csv_path = config["data"]["csv_path"]
    full_dataset = BrainNiiDataset(
        csv_file=csv_path,
        spatial_size=tuple(config["data"]["spatial_size"])
    )
    df = full_dataset.df

    # Raggruppamento per 'rid' per evitare che lo stesso paziente finisca in Train e Validation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config["training"]["seed"])
    train_idx, val_idx = next(gss.split(df, groups=df['rid']))

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    print(f"📊 Dataset: Totale {len(full_dataset)} | Train Set: {len(train_subset)} | Val Set: {len(val_subset)}")

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

    # 3. Inizializzazione Modello, Loss e Ottimizzatore
    model = Simple3DCNN(num_classes=config["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    # 4. Loop di Addestramento
    best_val_loss = float('inf')
    epochs = config["training"]["epochs"]
    logs = []

    print("\n🏋️ Inizio delle epoche di addestramento...")
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

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

        logs.append({
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })

        # Salva il checkpoint migliore
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{config['experiment_name']}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Modello migliore salvato in {checkpoint_path}! (Val Loss: {epoch_val_loss:.4f})")

    # Salva il file dei log finale
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(logs).to_csv(f"results/{config['experiment_name']}_logs.csv", index=False)
    print(f"\n✅ Addestramento completato! Log salvati in 'results/{config['experiment_name']}_logs.csv'.")

if __name__ == "__main__":
    train()