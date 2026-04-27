"""build_corpus.py — Module 1.

Build a multimodal Wikipedia-derived corpus laid out so it stresses the
two-tier (SSD hot / HDD cold) hierarchy used by the rest of the IMR-Fit
experiment.

Layout
------
    /mnt/hdd/wiki_corpus/
        text_shards/     128 MB shards of 512-token chunks (.jsonl)
        images/          raw image files, 100 KB – 5 MB each
        manifest.jsonl   one record per chunk: {chunk_id, kind, shard, offset, size}

    /mnt/ssd/faiss_index/
        index.faiss      IVF-Flat index (kept on hot tier — frequent random reads)
        ids.npy          aligned chunk_id array
        embeddings.npy   raw vectors (optional, for reranking)

Memory budget
-------------
    RAM    : <  2 GB   (streaming HF dataset, batched embedding, no in-memory corpus)
    VRAM   : <  1 GB   (sentence-transformers MiniLM-L6 on CPU by default)
    Disk   : ~ 20 GB   (default target; use --target-gb to scale)

Constraints honoured (per project spec)
---------------------------------------
* `datasets` library used in streaming mode, no `.shuffle()` / no full load.
* Embedding model: sentence-transformers/all-MiniLM-L6-v2 (CPU friendly).
* Embedding batch size capped at 64.
* FAISS index: IndexIVFFlat, NOT IndexFlatL2.
* Image sizes deliberately heterogeneous (100 KB–5 MB) so Z(b) is bimodal.

Fallback
--------
If `datasets` cannot reach huggingface.co (sandboxed runs, CI), the builder
falls back to a deterministic synthetic generator that produces:
    * pseudo-Wikipedia text drawn from a Zipf vocabulary
    * synthetic JPEG images with random RGB content
The downstream pipeline cannot tell the difference structurally — the
shapes, sizes and shard counts are identical.

Usage
-----
    python -m corpus.build_corpus --target-gb 4 --synthetic
    python -m corpus.build_corpus --target-gb 20 --hdd-root /mnt/hdd \\
                                  --ssd-root /mnt/ssd
"""

from __future__ import annotations

import io
import json
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Optional heavy deps — imported lazily so a CPU-only / sandbox run works.
try:  # pragma: no cover - exercised by integration test only
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:  # pragma: no cover
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


DEFAULT_HDD_ROOT = "/mnt/hdd/wiki_corpus"
DEFAULT_SSD_ROOT = "/mnt/ssd/faiss_index"

SHARD_SIZE_BYTES = 128 * 1024 * 1024     # 128 MB — matches IMR-Fit block size
CHUNK_TOKENS = 512                       # text chunk length in tokens (approx)
CHUNK_BYTES_AVG = 2 * 1024               # ~2 KB per chunk after tokenisation
EMBEDDING_DIM = 384                      # all-MiniLM-L6-v2
EMBED_BATCH = 64                         # capped per spec
IMG_MIN_BYTES = 100 * 1024
IMG_MAX_BYTES = 5 * 1024 * 1024
IMAGE_FRACTION = 0.20                    # 20 % of records carry an image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    """User-facing knobs for the corpus build."""

    hdd_root: str = DEFAULT_HDD_ROOT
    ssd_root: str = DEFAULT_SSD_ROOT
    target_size_gb: float = 20.0
    shard_size_bytes: int = SHARD_SIZE_BYTES
    chunk_tokens: int = CHUNK_TOKENS
    embed_batch: int = EMBED_BATCH
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_synthetic: bool = False              # True => skip HF download
    synthetic_seed: int = 0xC0FFEE
    huggingface_subset: str = "20231101.en"
    image_fraction: float = IMAGE_FRACTION
    log_every_chunks: int = 1000

    @property
    def text_shards_dir(self) -> Path:
        return Path(self.hdd_root) / "text_shards"

    @property
    def images_dir(self) -> Path:
        return Path(self.hdd_root) / "images"

    @property
    def manifest_path(self) -> Path:
        return Path(self.hdd_root) / "manifest.jsonl"

    @property
    def index_dir(self) -> Path:
        return Path(self.ssd_root)

    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        for p in (self.text_shards_dir, self.images_dir, self.index_dir):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Append-only JSONL shard writer with a 128 MB rollover policy.

    A shard corresponds to exactly one IMR-Fit block on the cold tier.  We
    write line-delimited JSON records and roll over to a new shard once the
    current file's size would exceed `max_bytes`.
    """

    def __init__(self, shards_dir: Path, max_bytes: int) -> None:
        self.shards_dir = shards_dir
        self.max_bytes = max_bytes
        self._idx = 0
        self._fh: Optional[io.BufferedWriter] = None
        self._current_path: Optional[Path] = None
        self._current_bytes = 0

    def _open_new_shard(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._idx = self._next_idx()
        self._current_path = self.shards_dir / f"shard_{self._idx:05d}.jsonl"
        self._fh = open(self._current_path, "wb", buffering=1024 * 1024)
        self._current_bytes = 0

    def _next_idx(self) -> int:
        existing = sorted(self.shards_dir.glob("shard_*.jsonl"))
        if not existing:
            return 0
        last = existing[-1].stem.split("_")[-1]
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)

    def write(self, record: Dict[str, Any]) -> Tuple[int, int]:
        """Append a record. Returns (shard_idx, byte_offset within shard)."""
        if self._fh is None:
            self._open_new_shard()
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if self._current_bytes + len(line) > self.max_bytes and self._current_bytes > 0:
            self._open_new_shard()
        offset = self._current_bytes
        assert self._fh is not None
        self._fh.write(line)
        self._current_bytes += len(line)
        return self._idx, offset

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Embedding helper (lazy import; falls back to deterministic hash vectors)
# ---------------------------------------------------------------------------


class _EmbeddingBackend:
    """Thin wrapper around sentence-transformers with a no-network fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:  # pragma: no cover
            print(
                f"[corpus] sentence-transformers unavailable ({exc}); "
                "falling back to deterministic hash embeddings.",
                file=sys.stderr,
            )
            self._fallback = True

    def encode(self, texts: List[str]) -> "np.ndarray":
        assert np is not None, "numpy is required"
        if not self._fallback and self._model is not None:
            vecs = self._model.encode(
                texts,
                batch_size=min(EMBED_BATCH, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.astype("float32")

        # Fallback: deterministic hash vectors (NOT semantically meaningful,
        # but sufficient to exercise the I/O pipeline).
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")
        for i, t in enumerate(texts):
            h = abs(hash(t))
            rng = random.Random(h)
            for j in range(EMBEDDING_DIM):
                out[i, j] = rng.uniform(-1.0, 1.0)
        # L2-normalise
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        out /= norms
        return out


# ---------------------------------------------------------------------------
# Streaming source: HuggingFace Wikipedia
# ---------------------------------------------------------------------------


def _load_hf_dataset_eagerly(subset: str):
    """Load a HuggingFace Wikipedia dataset object (NOT a generator).

    This function runs synchronously so any ImportError / RuntimeError is
    raised immediately — before a generator is created — making it safe to
    catch inside _open_stream's try/except.

    Returns the HF IterableDataset object on success.
    Raises RuntimeError (wrapping the root cause) on any failure.
    """
    # --- Step 1: check 'datasets' is installed ----------------------------
    try:
        from datasets import load_dataset  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"'datasets' package is not installed: {exc}\n"
            f"Fix: pip install 'datasets>=2.18'"
        ) from exc

    errors: list = []

    # Strategy 1: wikimedia/wikipedia (no dataset scripts, datasets >= 2.14)
    try:
        return load_dataset(
            "wikimedia/wikipedia", subset, split="train", streaming=True
        )
    except Exception as exc:
        errors.append(f"wikimedia/wikipedia/{subset}: {exc}")

    # Strategy 2: old 'wikipedia' package with explicit trust_remote_code
    try:
        return load_dataset(
            "wikipedia", subset, split="train",
            streaming=True, trust_remote_code=True,
        )
    except Exception as exc:
        errors.append(f"wikipedia (trust_remote_code=True): {exc}")

    # Strategy 3: old 'wikipedia' without flag (very old datasets versions)
    try:
        return load_dataset("wikipedia", subset, split="train", streaming=True)
    except Exception as exc:
        errors.append(f"wikipedia: {exc}")

    raise RuntimeError(
        "All Wikipedia loading strategies failed:\n"
        + "\n".join(f"  • {e}" for e in errors)
    )


def _hf_wikipedia_stream(subset: str) -> Iterator[Dict[str, Any]]:
    """Thin generator wrapper around _load_hf_dataset_eagerly.

    NOTE: this function must NOT be called directly from _open_stream —
    always go via _load_hf_dataset_eagerly so the ImportError is raised
    eagerly (not lazily inside the generator body where try/except can't
    catch it).
    """
    ds = _load_hf_dataset_eagerly(subset)   # raises on failure
    for row in ds:
        yield {"title": row.get("title", ""), "text": row.get("text", "")}


def _synthetic_wikipedia_stream(seed: int) -> Iterator[Dict[str, Any]]:
    """Deterministic, network-free fallback that mimics HF Wikipedia rows."""
    rng = random.Random(seed)
    vocab = [f"tok{i:04d}" for i in range(2000)]
    # Zipf-ish weights
    weights = [1.0 / (i + 1) for i in range(len(vocab))]
    article_idx = 0
    while True:
        article_idx += 1
        n_paragraphs = rng.randint(3, 30)
        paragraphs = []
        for _ in range(n_paragraphs):
            n_tokens = rng.randint(120, 800)
            tokens = rng.choices(vocab, weights=weights, k=n_tokens)
            paragraphs.append(" ".join(tokens))
        yield {
            "title": f"Synthetic Article {article_idx:06d}",
            "text": "\n\n".join(paragraphs),
        }


# ---------------------------------------------------------------------------
# Image generation / fetch
# ---------------------------------------------------------------------------


def _make_synthetic_image(target_bytes: int, seed: int) -> bytes:
    """Generate a JPEG of approximately `target_bytes` bytes.

    Uses Pillow if available, otherwise emits a JPEG-like blob filled with
    pseudo-random data (still produces a valid-on-disk file with the right
    size — accurate enough for the I/O simulator's purposes).
    """
    if _HAS_PIL and np is not None:
        # Pick a square dimension that yields roughly the requested size after
        # JPEG compression of random noise (random noise compresses ~1:1).
        side = max(64, int((target_bytes / 3) ** 0.5))
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, size=(side, side, 3), dtype="uint8")
        img = Image.fromarray(arr, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
        if len(data) < target_bytes:
            data = data + os.urandom(target_bytes - len(data))
        return data[:target_bytes]

    rng = random.Random(seed)
    # JPEG SOI/EOI markers so the file is at least sniffable.
    head = b"\xff\xd8\xff\xe0"
    tail = b"\xff\xd9"
    body_len = max(0, target_bytes - len(head) - len(tail))
    body = bytes(rng.getrandbits(8) for _ in range(body_len))
    return head + body + tail


def _sample_image_bytes(rng: random.Random) -> int:
    """Bimodal size sampler — small thumbnails or large hero images.

    This produces the bimodal Z(b) distribution called out in the spec.
    """
    if rng.random() < 0.6:
        # thumbnail-ish: 100 KB – 400 KB
        return rng.randint(IMG_MIN_BYTES, 400 * 1024)
    # hero image: 1 MB – 5 MB
    return rng.randint(1 * 1024 * 1024, IMG_MAX_BYTES)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


@dataclass
class _BuildState:
    bytes_written: int = 0
    chunks_written: int = 0
    images_written: int = 0
    started_at: float = field(default_factory=time.monotonic)


class CorpusBuilder:
    """Driver that streams a Wikipedia source into shards + a FAISS index."""

    def __init__(self, config: Optional[CorpusConfig] = None) -> None:
        self.cfg = config or CorpusConfig()
        self.cfg.ensure_dirs()
        self._shard_writer = ShardWriter(self.cfg.text_shards_dir,
                                         self.cfg.shard_size_bytes)
        self._embedder = _EmbeddingBackend(self.cfg.embedding_model)
        self._state = _BuildState()
        self._rng = random.Random(self.cfg.synthetic_seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        target_bytes = int(self.cfg.target_size_gb * (1024 ** 3))
        manifest = open(self.cfg.manifest_path, "w", encoding="utf-8")
        try:
            stream = self._open_stream()
            batch_texts: List[str] = []
            batch_records: List[Dict[str, Any]] = []
            for chunk_record in self._chunkify(stream):
                batch_texts.append(chunk_record["text"])
                batch_records.append(chunk_record)

                if len(batch_texts) >= self.cfg.embed_batch:
                    self._flush_batch(batch_texts, batch_records, manifest)
                    batch_texts.clear()
                    batch_records.clear()

                if self._state.bytes_written >= target_bytes:
                    break

                if (self._state.chunks_written > 0 and
                        self._state.chunks_written %
                        self.cfg.log_every_chunks == 0):
                    self._log_progress()

            if batch_texts:
                self._flush_batch(batch_texts, batch_records, manifest)

        finally:
            manifest.close()
            self._shard_writer.close()

        # Build the FAISS index from saved embeddings (deferred — keeps RAM low).
        index_meta = self._build_faiss_index()

        return {
            "bytes_written": self._state.bytes_written,
            "chunks": self._state.chunks_written,
            "images": self._state.images_written,
            "shards": self._count_shards(),
            "elapsed_sec": time.monotonic() - self._state.started_at,
            "index": index_meta,
        }

    # ------------------------------------------------------------------
    # Stream selection
    # ------------------------------------------------------------------

    def _open_stream(self) -> Iterator[Dict[str, Any]]:
        if self.cfg.use_synthetic:
            return _synthetic_wikipedia_stream(self.cfg.synthetic_seed)
        # _load_hf_dataset_eagerly runs synchronously so any ImportError /
        # RuntimeError is raised *here* (not lazily inside a generator body).
        try:
            ds = _load_hf_dataset_eagerly(self.cfg.huggingface_subset)

            def _wrap():
                for row in ds:
                    yield {"title": row.get("title", ""),
                           "text": row.get("text", "")}

            return _wrap()
        except Exception as exc:
            print(
                f"[corpus] HF dataset unavailable — {exc}\n"
                f"[corpus] Switching to synthetic fallback "
                f"(add --synthetic to silence this warning).",
                file=sys.stderr,
            )
            return _synthetic_wikipedia_stream(self.cfg.synthetic_seed)

    # ------------------------------------------------------------------
    # Article -> 512-token chunks
    # ------------------------------------------------------------------

    def _chunkify(
        self,
        stream: Iterator[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        for art_idx, article in enumerate(stream):
            tokens = article["text"].split()
            if not tokens:
                continue
            n_chunks = max(1, len(tokens) // self.cfg.chunk_tokens)
            for ci in range(n_chunks):
                start = ci * self.cfg.chunk_tokens
                end = min(start + self.cfg.chunk_tokens, len(tokens))
                if end - start < 32:    # skip near-empty tail chunks
                    continue
                chunk_text = " ".join(tokens[start:end])
                chunk_id = f"wiki_{art_idx:06d}_{ci:04d}"
                rec = {
                    "chunk_id": chunk_id,
                    "title": article["title"],
                    "text": chunk_text,
                    "kind": "text",
                }
                # Maybe attach an image
                if self._rng.random() < self.cfg.image_fraction:
                    img_path = self._emit_image(chunk_id)
                    rec["image_path"] = str(img_path)
                yield rec

    # ------------------------------------------------------------------
    # Image emission
    # ------------------------------------------------------------------

    def _emit_image(self, chunk_id: str) -> Path:
        target = _sample_image_bytes(self._rng)
        # Per-chunk seed so re-runs are deterministic.
        data = _make_synthetic_image(target, abs(hash(chunk_id)) & 0xFFFFFFFF)
        # Shard images by first 3 chars to keep dirs fan-out friendly.
        prefix = chunk_id[:8]
        out_dir = self.cfg.images_dir / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{chunk_id}.jpg"
        out_path.write_bytes(data)
        self._state.images_written += 1
        self._state.bytes_written += len(data)
        return out_path

    # ------------------------------------------------------------------
    # Batch flush: embed + write shards + manifest
    # ------------------------------------------------------------------

    def _flush_batch(
        self,
        texts: List[str],
        records: List[Dict[str, Any]],
        manifest: Any,
    ) -> None:
        embeddings = self._embedder.encode(texts)
        for rec, vec in zip(records, embeddings):
            shard_idx, offset = self._shard_writer.write(rec)
            manifest_row = {
                "chunk_id": rec["chunk_id"],
                "kind": "image" if "image_path" in rec else "text",
                "shard": shard_idx,
                "offset": offset,
                "size": len(json.dumps(rec, ensure_ascii=False).encode("utf-8")),
                "image_path": rec.get("image_path"),
            }
            manifest.write(json.dumps(manifest_row) + "\n")
            self._state.chunks_written += 1
            self._state.bytes_written += manifest_row["size"]
            self._cache_embedding(rec["chunk_id"], vec)

    def _cache_embedding(self, chunk_id: str, vec: "np.ndarray") -> None:
        # Embeddings are tiny (1.5 KB each at fp32) — accumulating in memory
        # is fine for ≤ 1M chunks (~1.5 GB).  For larger runs we'd spill to a
        # memmap; not needed at the 20 GB target.
        if not hasattr(self, "_emb_buf"):
            self._emb_buf: List["np.ndarray"] = []
            self._emb_ids: List[str] = []
        self._emb_buf.append(vec)
        self._emb_ids.append(chunk_id)

    # ------------------------------------------------------------------
    # FAISS index (IndexIVFFlat — cheap on RAM)
    # ------------------------------------------------------------------

    def _build_faiss_index(self) -> Dict[str, Any]:
        if not hasattr(self, "_emb_buf") or not self._emb_buf:
            return {"status": "no-embeddings"}
        assert np is not None
        embeddings = np.stack(self._emb_buf)
        ids = np.array(self._emb_ids)

        np.save(self.cfg.index_dir / "embeddings.npy", embeddings)
        np.save(self.cfg.index_dir / "ids.npy", ids)

        try:
            import faiss  # type: ignore
        except Exception as exc:
            return {"status": f"faiss-unavailable: {exc}",
                    "embeddings_path": str(self.cfg.index_dir / "embeddings.npy")}

        d = embeddings.shape[1]
        n = embeddings.shape[0]
        nlist = max(1, min(4096, int(n ** 0.5)))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        # IVF needs training; for very small corpora train on the data itself.
        train_n = min(n, max(nlist * 39, nlist + 1))
        index.train(embeddings[:train_n])
        index.add(embeddings)
        index.nprobe = min(16, nlist)

        idx_path = self.cfg.index_dir / "index.faiss"
        faiss.write_index(index, str(idx_path))
        return {
            "status": "ok",
            "vectors": int(n),
            "dim": int(d),
            "nlist": int(nlist),
            "index_path": str(idx_path),
        }

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def _count_shards(self) -> int:
        return len(list(self.cfg.text_shards_dir.glob("shard_*.jsonl")))

    def _log_progress(self) -> None:
        gb = self._state.bytes_written / (1024 ** 3)
        secs = time.monotonic() - self._state.started_at
        rate = gb / secs if secs > 0 else 0
        print(
            f"[corpus] chunks={self._state.chunks_written} "
            f"images={self._state.images_written} "
            f"size={gb:.2f} GB rate={rate*1024:.1f} MB/s",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hdd-root", default=DEFAULT_HDD_ROOT,
                   help="Cold-tier corpus root (default: %(default)s)")
    p.add_argument("--ssd-root", default=DEFAULT_SSD_ROOT,
                   help="Hot-tier index root (default: %(default)s)")
    p.add_argument("--target-gb", type=float, default=20.0,
                   help="Target corpus size in gigabytes")
    p.add_argument("--shard-mb", type=int, default=128,
                   help="Shard size in megabytes (default 128, matches block_size)")
    p.add_argument("--embed-batch", type=int, default=EMBED_BATCH,
                   help=f"Embedding batch size (default {EMBED_BATCH})")
    p.add_argument("--hf-subset", default="20231101.en",
                   help="HuggingFace Wikipedia config (default: %(default)s)."
                        " Loaded via wikimedia/wikipedia first, then "
                        "wikipedia with trust_remote_code as fallback.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use deterministic synthetic source (offline)")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="RNG seed for synthetic mode")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = CorpusConfig(
        hdd_root=args.hdd_root,
        ssd_root=args.ssd_root,
        target_size_gb=args.target_gb,
        shard_size_bytes=args.shard_mb * 1024 * 1024,
        embed_batch=args.embed_batch,
        use_synthetic=args.synthetic,
        synthetic_seed=args.seed,
        huggingface_subset=args.hf_subset,
    )
    builder = CorpusBuilder(cfg)
    summary = builder.build()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
