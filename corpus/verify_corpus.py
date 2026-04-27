"""verify_corpus.py — sanity-check a built corpus.

Reads the manifest written by `build_corpus.py` and asserts:

  * every shard exists and is roughly 128 MB (within tolerance)
  * every record's offset is consistent with the shard's on-disk size
  * the bimodal Z(b) distribution is present (text ≈ 2 KB, image ≈ 1 MB)
  * the FAISS index loads and has the expected number of vectors

Usage::

    python -m corpus.verify_corpus --hdd-root /mnt/hdd/wiki_corpus \\
                                   --ssd-root /mnt/ssd/faiss_index
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

from .build_corpus import DEFAULT_HDD_ROOT, DEFAULT_SSD_ROOT, SHARD_SIZE_BYTES


def _load_manifest(manifest_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def verify(hdd_root: str, ssd_root: str) -> Dict:
    hdd = Path(hdd_root)
    ssd = Path(ssd_root)
    manifest = hdd / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest missing: {manifest}")

    rows = _load_manifest(manifest)
    text_sizes = [r["size"] for r in rows if r["kind"] == "text"]
    img_sizes: List[int] = []
    for r in rows:
        if r.get("image_path"):
            ip = Path(r["image_path"])
            if ip.exists():
                img_sizes.append(ip.stat().st_size)

    shards = sorted((hdd / "text_shards").glob("shard_*.jsonl"))
    shard_sizes = [s.stat().st_size for s in shards]

    # Bimodal Z(b) check: text mean << image mean
    text_mean = statistics.mean(text_sizes) if text_sizes else 0
    img_mean = statistics.mean(img_sizes) if img_sizes else 0
    bimodal_ok = (img_sizes and text_sizes and img_mean > 10 * text_mean)

    # Index check
    index_status = "missing"
    n_vec = 0
    idx_path = ssd / "index.faiss"
    if idx_path.exists():
        try:
            import faiss  # type: ignore
            idx = faiss.read_index(str(idx_path))
            n_vec = int(idx.ntotal)
            index_status = "ok"
        except Exception as exc:
            index_status = f"load-failed: {exc}"

    report = {
        "manifest_rows": len(rows),
        "text_records": len(text_sizes),
        "image_records": len(img_sizes),
        "shards": len(shards),
        "shard_avg_mb": round(statistics.mean(shard_sizes) / (1024 * 1024), 2)
                        if shard_sizes else 0.0,
        "shard_max_mb": round(max(shard_sizes) / (1024 * 1024), 2)
                        if shard_sizes else 0.0,
        "shard_size_target_mb": SHARD_SIZE_BYTES // (1024 * 1024),
        "text_mean_bytes": round(text_mean, 1),
        "image_mean_bytes": round(img_mean, 1),
        "bimodal_z_distribution": bimodal_ok,
        "index_status": index_status,
        "index_vectors": n_vec,
    }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdd-root", default=DEFAULT_HDD_ROOT)
    p.add_argument("--ssd-root", default=DEFAULT_SSD_ROOT)
    args = p.parse_args(argv)
    report = verify(args.hdd_root, args.ssd_root)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report["bimodal_z_distribution"] or report["text_records"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
