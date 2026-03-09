"""
synthetic_cv.py - Synthetic ImageNet-style dataset generator.

Generates a fake image classification dataset under a configurable root
directory (default: /mnt/imrsim/).  Each image is a random JPEG of ~100 KB
to mimic real CV training data sizes.

Usage (CLI)::

    python workloads/synthetic_cv.py \\
        --root /mnt/imrsim/imagenet \\
        --classes 10 \\
        --images-per-class 100 \\
        --dry-run

Usage (Python)::

    from workloads.synthetic_cv import generate_dataset, SyntheticCVConfig
    cfg = SyntheticCVConfig(root="/mnt/imrsim/imagenet", n_classes=10, m_images=100)
    paths = generate_dataset(cfg)
"""

from __future__ import annotations

import argparse
import io
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticCVConfig:
    root: str = "/mnt/imrsim/imagenet"
    n_classes: int = 10
    m_images: int = 100
    target_size_bytes: int = 100 * 1024  # ~100 KB per image
    image_width: int = 224
    image_height: int = 224
    seed: int = 42
    dry_run: bool = False
    verbose: bool = True


# ---------------------------------------------------------------------------
# JPEG generation
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int, height: int, target_bytes: int, rng: random.Random) -> bytes:
    """
    Create a minimal valid JPEG filled with random colour noise.

    We use PIL/Pillow when available for realistic file sizes; otherwise we
    produce a raw JPEG skeleton that most decoders accept.
    """
    try:
        from PIL import Image
        import numpy as np

        arr = np.array(
            [rng.randint(0, 255) for _ in range(width * height * 3)],
            dtype=np.uint8,
        ).reshape((height, width, 3))
        img = Image.fromarray(arr, "RGB")

        buf = io.BytesIO()
        # Adjust quality so we get close to target_bytes
        quality = _estimate_quality(width, height, target_bytes)
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    except ImportError:
        # Fallback: write a minimal JPEG + random padding to reach target size
        return _minimal_jpeg(width, height, target_bytes, rng)


def _estimate_quality(width: int, height: int, target_bytes: int) -> int:
    """Rough heuristic: quality ~ 10 gives ~0.5 bytes/pixel for 224x224."""
    pixels = width * height
    bpp_target = target_bytes / pixels
    # quality ≈ 200 * bpp for typical photographic content (empirical)
    quality = int(bpp_target * 200)
    return max(5, min(95, quality))


def _minimal_jpeg(width: int, height: int, target_bytes: int, rng: random.Random) -> bytes:
    """
    Produce a syntactically valid but minimal JPEG byte sequence.

    SOI + APP0 + comment padding + EOI.
    This won't decode to a real image but will exercise file I/O.
    """
    soi = b"\xff\xd8"
    app0 = (
        b"\xff\xe0"  # APP0 marker
        b"\x00\x10"  # length = 16
        b"JFIF\x00"  # identifier
        b"\x01\x01"  # version
        b"\x00"      # aspect ratio units
        b"\x00\x01\x00\x01"  # X/Y density
        b"\x00\x00"  # thumbnail size
    )
    eoi = b"\xff\xd9"
    overhead = len(soi) + len(app0) + len(eoi) + 4  # +4 for comment header
    padding_size = max(0, target_bytes - overhead)
    comment = (
        b"\xff\xfe"  # COM marker
        + struct.pack(">H", padding_size + 2)
        + bytes(rng.randint(0, 255) for _ in range(padding_size))
    )
    return soi + app0 + comment + eoi


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(cfg: SyntheticCVConfig) -> List[str]:
    """
    Generate the synthetic dataset according to cfg.

    Returns a list of created file paths.

    When cfg.dry_run is True, only prints what would be created.
    """
    rng = random.Random(cfg.seed)
    root = Path(cfg.root)

    if cfg.verbose:
        print(f"[synthetic_cv] Generating dataset at {root}")
        print(f"  classes={cfg.n_classes}, images/class={cfg.m_images}, "
              f"~{cfg.target_size_bytes // 1024} KB/image")
        if cfg.dry_run:
            print("  [DRY RUN] No files will be written.")

    created: List[str] = []
    total_files = cfg.n_classes * cfg.m_images
    start = time.monotonic()

    for cls_idx in range(cfg.n_classes):
        class_name = f"class_{cls_idx:04d}"
        class_dir = root / class_name

        if not cfg.dry_run:
            class_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(cfg.m_images):
            img_path = class_dir / f"{class_name}_{img_idx:05d}.jpg"

            if cfg.dry_run:
                created.append(str(img_path))
                continue

            if img_path.exists():
                created.append(str(img_path))
                continue

            jpeg_data = _make_jpeg_bytes(
                cfg.image_width, cfg.image_height, cfg.target_size_bytes, rng
            )
            img_path.write_bytes(jpeg_data)
            created.append(str(img_path))

            if cfg.verbose and (len(created) % 100 == 0 or len(created) == total_files):
                elapsed = time.monotonic() - start
                print(f"  {len(created)}/{total_files} files "
                      f"({elapsed:.1f}s)", end="\r", flush=True)

    if cfg.verbose:
        elapsed = time.monotonic() - start
        total_mb = sum(
            os.path.getsize(p) for p in created if os.path.exists(p)
        ) / (1024 ** 2)
        print(f"\n[synthetic_cv] Done: {len(created)} files, "
              f"{total_mb:.1f} MB total, {elapsed:.1f}s elapsed")

    return created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> SyntheticCVConfig:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic ImageNet-style dataset for IMR-Fit experiments."
    )
    parser.add_argument("--root", default="/mnt/imrsim/imagenet",
                        help="Root directory for the dataset (default: /mnt/imrsim/imagenet)")
    parser.add_argument("--classes", type=int, default=10, dest="n_classes",
                        help="Number of classes (default: 10)")
    parser.add_argument("--images-per-class", type=int, default=100, dest="m_images",
                        help="Images per class (default: 100)")
    parser.add_argument("--size-kb", type=int, default=100,
                        help="Target JPEG size in KB (default: 100)")
    parser.add_argument("--width", type=int, default=224, help="Image width px (default: 224)")
    parser.add_argument("--height", type=int, default=224, help="Image height px (default: 224)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args(argv)
    return SyntheticCVConfig(
        root=args.root,
        n_classes=args.n_classes,
        m_images=args.m_images,
        target_size_bytes=args.size_kb * 1024,
        image_width=args.width,
        image_height=args.height,
        seed=args.seed,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    paths = generate_dataset(cfg)
    sys.exit(0)
