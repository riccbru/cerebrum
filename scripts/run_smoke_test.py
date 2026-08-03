import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, set_seed
from src.dataset import BrainNiiDataset
from src.models import Simple3DCNN

def main():
    print("🚀 Running Smoke Test...")
    
    # 1. Setup
    config = load_config("configs/config_adni.yaml")
    set_seed(config["training"]["seed"])
    
    # 2. Load dataset
    dataset = BrainNiiDataset(
        csv_file=config["data"]["csv_path"],
        spatial_size=tuple(config["data"]["spatial_size"])
    )
    
    # 3. Test on one sample
    img, label = dataset[0]
    model = Simple3DCNN(num_classes=config["model"]["num_classes"])
    out = model(img.unsqueeze(0))
    
    print(f"✅ Shape Tensor: {img.shape} | Out shape: {out.shape}")

if __name__ == "__main__":
    main()
