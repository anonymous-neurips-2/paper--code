# src/datasets/fruits360.py

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.tv_datasets.common import GenericDataset


class Fruits360(GenericDataset):
    def __init__(self, preprocess, location, batch_size=128, num_workers=4):
        super().__init__()

        fruit_root = os.path.join(location, "Fruits360")
        if not os.path.isdir(fruit_root):
            raise FileNotFoundError(f"There is no `Fruits360` directory: {fruit_root}")

        candidates = [fruit_root] + [
            os.path.join(fruit_root, d)
            for d in os.listdir(fruit_root)
            if os.path.isdir(os.path.join(fruit_root, d))
        ]

        base_dir = None

        for c in candidates:
            if os.path.isdir(os.path.join(c, "Training")) and os.path.isdir(os.path.join(c, "Test")):
                base_dir = c
                break

        if base_dir is None:
            for c in candidates:
                for d in os.listdir(c):
                    nested = os.path.join(c, d)
                    if not os.path.isdir(nested):
                        continue
                    if os.path.isdir(os.path.join(nested, "Training")) and os.path.isdir(os.path.join(nested, "Test")):
                        base_dir = nested
                        break
                if base_dir:
                    break

        if base_dir is None:
            msg  = "There is no `Training/Test` directory. candidates:\n"
            msg += "\n".join(candidates)
            raise FileNotFoundError(msg)

        train_dir = os.path.join(base_dir, "Training")
        test_dir  = os.path.join(base_dir, "Test")

        self.train_dataset = ImageFolder(train_dir, transform=preprocess)
        self.test_dataset  = ImageFolder(test_dir,  transform=preprocess)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.classnames = self.train_dataset.classes
