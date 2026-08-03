import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    ResizeWithPadOrCropd
)

class BrainNiiDataset(Dataset):
    def __init__(self, csv_file, spatial_size=(160, 190, 160)):
        if not os.path.exists(csv_file):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            alt_path = os.path.join(project_root, csv_file)
            if os.path.exists(alt_path):
                csv_file = alt_path

        self.df = pd.read_csv(csv_file)
        self.label_map = {1: 0, 2: 1, 3: 2} # AD:0, CN:1, MCI:2
        
        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityd(keys=["image"]),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_dict = {"image": row['filepath']}
        data = self.transforms(file_dict)
        
        raw_label = int(row['diagnosis'])
        mapped_label = self.label_map.get(raw_label, raw_label)
        
        return data["image"], torch.tensor(mapped_label, dtype=torch.long)
