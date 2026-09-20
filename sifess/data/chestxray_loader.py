"""NIH ChestX-ray14 loader stub for full-scale runs (documented, optional).

Day-0 training uses ChestMNIST. Point ``root`` at a local NIH CXR directory
with the standard layout when running full experiments on Kaggle / cluster.

Expected layout (common Kaggle NIH mirror)::

    root/
      images_001/images/*.png
      ...
      Data_Entry_2017.csv

This module does not download NIH data (licensing / size). See docs/EXPERIMENT_CARD.md.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


NIH_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


class NIHCXRDataset(Dataset):
    """Minimal multi-label NIH CXR reader. Raises FileNotFoundError if missing."""

    def __init__(
        self,
        root: str,
        split_csv: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.root = root
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        entry = os.path.join(root, "Data_Entry_2017.csv")
        if not os.path.isfile(entry):
            raise FileNotFoundError(
                f"NIH CXR metadata not found at {entry}. "
                "Day-0 uses ChestMNIST; place NIH files locally for full runs."
            )
        df = pd.read_csv(entry)
        if split_csv and os.path.isfile(split_csv):
            keep = set(pd.read_csv(split_csv, header=None)[0].astype(str).tolist())
            df = df[df["Image Index"].astype(str).isin(keep)]
        self.df = df.reset_index(drop=True)
        self.findings = NIH_FINDINGS

    def __len__(self) -> int:
        return len(self.df)

    def _find_image(self, name: str) -> str:
        # Search common folder patterns
        for dirpath, _, files in os.walk(self.root):
            if name in files:
                return os.path.join(dirpath, name)
        raise FileNotFoundError(name)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        name = str(row["Image Index"])
        path = self._find_image(name)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        labels = torch.zeros(len(self.findings), dtype=torch.float32)
        raw = str(row["Finding Labels"])
        if raw != "No Finding":
            for f in raw.split("|"):
                if f in self.findings:
                    labels[self.findings.index(f)] = 1.0
        return {"image": x, "label": labels, "id": name}


def build_nih_cxr(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    split_csv: Optional[str] = None,
) -> Tuple[Dataset, DataLoader]:
    ds = NIHCXRDataset(root=root, split_csv=split_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return ds, loader
