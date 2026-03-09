"""
dataloader_factory.py - Factory for creating (optionally profiled) DataLoaders.

Supports both a real torchvision ImageFolder dataset and a lightweight
mock dataset that requires no GPU / torchvision install, for dry-run testing.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Lightweight mock dataset (no torchvision required)
# ---------------------------------------------------------------------------

class MockImageDataset:
    """
    A minimal map-style dataset that scans an ImageFolder-style directory.

    Returns (filepath, class_index) tuples.  Useful for profiling without
    actually decoding images.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []
        self._scan(root)

    def _scan(self, root: str) -> None:
        if not os.path.isdir(root):
            return
        class_dirs = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.classes = class_dirs
        for cls_idx, cls_name in enumerate(class_dirs):
            cls_dir = os.path.join(root, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]  # (path, class_idx)


class MockDataLoader:
    """
    Bare-bones DataLoader that yields batches of (path, label) tuples
    without any tensor conversion.  Used for dry-run / no-GPU environments.
    """

    def __init__(self, dataset: MockImageDataset, batch_size: int = 32) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        samples = self.dataset.samples
        for start in range(0, len(samples), self.batch_size):
            batch = samples[start: start + self.batch_size]
            paths = [s[0] for s in batch]
            labels = [s[1] for s in batch]
            yield paths, labels


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_dataloader(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    use_mock: bool = False,
    transform: Optional[Callable] = None,
) -> Any:
    """
    Create a DataLoader for the dataset at data_root.

    Args:
        data_root: path to an ImageFolder-style directory
        batch_size: samples per batch
        num_workers: worker processes (ignored for mock)
        shuffle: whether to shuffle (ignored for mock)
        use_mock: if True, return a MockDataLoader (no torch required)
        transform: torchvision transform (ignored for mock)

    Returns:
        A torch DataLoader or MockDataLoader.
    """
    if use_mock:
        dataset = MockImageDataset(data_root)
        return MockDataLoader(dataset, batch_size=batch_size)

    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms as T

        if transform is None:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        dataset = datasets.ImageFolder(root=data_root, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    except ImportError as exc:
        raise ImportError(
            "torch/torchvision not installed. "
            "Use use_mock=True for dry-run mode."
        ) from exc


def make_profiled_dataloader(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    use_mock: bool = False,
    mount_point: str = "/mnt/imrsim",
    recency_lambda: float = 1.0,
) -> Any:
    """
    Create a DataLoaderProfiler wrapping the base DataLoader.

    Returns a DataLoaderProfiler (or MockDataLoader if use_mock=True and
    the profiler cannot be attached).
    """
    from imrfit.profiler import DataLoaderProfiler

    base = make_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        use_mock=use_mock,
    )
    return DataLoaderProfiler(
        base,
        mount_point=mount_point,
        recency_lambda=recency_lambda,
    )
