"""ChestMNIST (MedMNIST) loaders for Day-0 smoke / SSL.

Default image size 224. Multi-crop: 2 global + 4 local (DINO-style).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _ensure_medmnist():
    try:
        import medmnist
        from medmnist import ChestMNIST, INFO
    except ImportError as e:
        raise ImportError("medmnist is required: pip install medmnist") from e
    return medmnist, ChestMNIST, INFO


class MultiCropChestMNIST(Dataset):
    """Wraps ChestMNIST with DINO multi-crop views + labels for probing."""

    def __init__(
        self,
        split: str = "train",
        size: int = 224,
        download: bool = True,
        n_global: int = 2,
        n_local: int = 4,
        global_size: int = 224,
        local_size: int = 96,
        as_rgb: bool = True,
    ):
        _, ChestMNIST, INFO = _ensure_medmnist()
        info = INFO["chestmnist"]
        self.n_classes = len(info["label"])
        self.dataset = ChestMNIST(split=split, download=download, size=size, as_rgb=as_rgb)
        self.n_global = n_global
        self.n_local = n_local

        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        self.global_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.local_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.eval_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(global_size),
                transforms.CenterCrop(global_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def _hwc_uint8(self, img: Any) -> np.ndarray:
        if isinstance(img, np.ndarray):
            x = img
        else:
            x = np.array(img)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return x.astype(np.uint8)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        # medmnist may return PIL or ndarray depending on version
        arr = self._hwc_uint8(img)
        label = torch.as_tensor(label).float().view(-1)
        if self.split == "train" and (self.n_global + self.n_local) > 0:
            crops = [self.global_tf(arr) for _ in range(self.n_global)]
            crops += [self.local_tf(arr) for _ in range(self.n_local)]
            return {"crops": crops, "label": label, "image": self.eval_tf(arr)}
        return {"image": self.eval_tf(arr), "label": label}


def chestmnist_collate_multicrop(batch: List[Dict]) -> Dict[str, Any]:
    if "crops" in batch[0]:
        n_crops = len(batch[0]["crops"])
        crops = [
            torch.stack([b["crops"][i] for b in batch], dim=0) for i in range(n_crops)
        ]
        labels = torch.stack([b["label"] for b in batch], dim=0)
        images = torch.stack([b["image"] for b in batch], dim=0)
        return {"crops": crops, "label": labels, "image": images}
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"image": images, "label": labels}


def build_chestmnist(
    split: str = "train",
    size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    multicrop: bool = True,
    n_global: int = 2,
    n_local: int = 4,
    subset: Optional[int] = None,
) -> Tuple[Dataset, DataLoader]:
    ds = MultiCropChestMNIST(
        split=split,
        size=size,
        download=download,
        n_global=n_global if multicrop and split == "train" else 0,
        n_local=n_local if multicrop and split == "train" else 0,
    )
    if subset is not None:
        indices = list(range(min(subset, len(ds))))
        ds = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=chestmnist_collate_multicrop,
        drop_last=(split == "train"),
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader
