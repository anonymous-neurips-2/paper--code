# src/datasets/<파일명>.py
import os, torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from src.tv_datasets.common import GenericDataset


class Weather(GenericDataset):
    def __init__(self, preprocess, location, batch_size=128, num_workers=4):
        super().__init__()
        root = os.path.join(location, "Weather", "dataset")   # ← 폴더명만 교체
        full = ImageFolder(root, transform=preprocess)

        # 10 % validation split
        val_len   = max(1, int(0.1 * len(full)))
        train_len = len(full) - val_len
        train_set, val_set = random_split(
            full, [train_len, val_len],
            generator=torch.Generator().manual_seed(0)
        )

        self.train_dataset = train_set
        self.test_dataset  = val_set
        self.train_loader  = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=False
        )
        self.test_loader   = DataLoader(
            val_set,  batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=False
        )
        self.classnames = full.classes
