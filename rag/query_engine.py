"""query_engine.py — Module 3 (LLM-side).

A minimal RAG engine that:
    1. Embeds a query (CPU MiniLM-L6-v2 — same model used to build the index).
    2. Retrieves top-k chunk_ids from the FAISS index on /mnt/ssd.
    3. Reads each retrieved chunk via the TieredStorageSimulator
       (this is what generates the cold-tier trace).
    4. Optionally feeds the concatenated context to Qwen2-VL-2B-Instruct-Q4
       via llama-cpp-python, with CUDA off-load capped at 28 layers.
    5. Returns a QueryResult.

VRAM budget (Qwen2-VL-2B-Q4_K_M)
--------------------------------
    Model weights         ~ 1.6 GB  (28 layers off-loaded -> ~ 4 GB VRAM)
    KV cache (n_ctx=2048) ~ 2 GB
    Total                 ~ 6 GB    (leaves ~ 2 GB headroom on a GTX 1080)

Fallbacks
---------
* `llama-cpp-python` not built / no GPU
        -> use a tiny `transformers` Q4 model (bitsandbytes) if available,
           else just skip generation and return retrieved-context-only.
* `faiss` missing
        -> brute-force cosine search on the saved embeddings.npy.
* `sentence-transformers` missing
        -> deterministic hash embeddings (matches build_corpus fallback).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from storage.tier_simulator import TieredStorageSimulator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Engine-wide knobs.  All paths default to the spec'd mount points."""
    ssd_index_root: str = "/mnt/ssd/faiss_index"
    model_path: str = "/mnt/ssd/models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    n_ctx: int = 2048           # capped per spec
    n_gpu_layers: int = 28      # 28 of 28 layers off-loaded to GPU
    n_threads: int = 4
    max_tokens: int = 128

    top_k: int = 5
    skip_llm: bool = False      # True -> pure retrieval, no generation
    seed: int = 0xBEEF


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    query: str
    retrieved_chunks: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    bytes_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    latency_ms: float = 0.0
    answer: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_chunks": self.retrieved_chunks,
            "retrieved_scores": self.retrieved_scores,
            "bytes_read": self.bytes_read,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "latency_ms": self.latency_ms,
            "answer": self.answer,
        }


# ---------------------------------------------------------------------------
# Embedder (mirrors build_corpus' fallback path)
# ---------------------------------------------------------------------------


class _QueryEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:  # pragma: no cover
            print(
                f"[rag] sentence-transformers unavailable ({exc}); "
                "using deterministic hash embeddings.",
                file=sys.stderr,
            )
            self._fallback = True

    def encode(self, query: str, dim: int = 384):
        """Return a unit-norm embedding vector (numpy array if available,
        plain list otherwise).  Never raises — always returns something."""
        if not self._fallback and self._model is not None:
            # sentence-transformers path (returns numpy array)
            v = self._model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True)
            return v.astype("float32")[0]

        # Hash-based deterministic fallback — works with or without numpy.
        rng = random.Random(abs(hash(query)))
        floats = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        norm = sum(x * x for x in floats) ** 0.5 + 1e-9
        floats = [x / norm for x in floats]

        if np is not None:
            return np.array(floats, dtype="float32")
        return floats   # plain list — FAISS won't work but trace still records


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class _Retriever:
    """FAISS-backed top-k retriever with brute-force fallback."""

    def __init__(self, ssd_root: str) -> None:
        self.ssd_root = Path(ssd_root)
        self._faiss_index = None
        self._brute_embeddings: Optional["np.ndarray"] = None
        self._ids: Optional["np.ndarray"] = None
        self._load()

    def _load(self) -> None:
        ids_path = self.ssd_root / "ids.npy"
        emb_path = self.ssd_root / "embeddings.npy"
        idx_path = self.ssd_root / "index.faiss"

        if not ids_path.exists():
            print(
                f"[rag] FAISS index not found at {ids_path}.\n"
                f"[rag] Run corpus build first: "
                f"python -m corpus.build_corpus --target-gb <N>",
                file=sys.stderr,
            )
            return

        if np is None:
            print("[rag] numpy not installed — retriever disabled", file=sys.stderr)
            return
        self._ids = np.load(ids_path, allow_pickle=True)

        if idx_path.exists():
            try:
                import faiss  # type: ignore
                self._faiss_index = faiss.read_index(str(idx_path))
                return
            except Exception as exc:
                print(f"[rag] faiss unavailable ({exc}); using brute-force",
                      file=sys.stderr)

        if emb_path.exists():
            self._brute_embeddings = np.load(emb_path)

    def search(self, q_vec, top_k: int) -> List[Tuple[str, float]]:
        if self._ids is None or np is None:
            return []

        if not hasattr(q_vec, "reshape"):
            q_vec = np.array(q_vec, dtype="float32")

        if self._faiss_index is not None:
            D, I = self._faiss_index.search(q_vec.reshape(1, -1), top_k)
            scores = D[0].tolist()
            indices = I[0].tolist()
        elif self._brute_embeddings is not None:
            sims = self._brute_embeddings @ q_vec
            indices = np.argsort(-sims)[:top_k].tolist()
            scores = sims[indices].tolist()
        else:
            return []

        out: List[Tuple[str, float]] = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self._ids):
                out.append((str(self._ids[idx]), float(score)))
        return out

    def random_chunk_ids(self, n: int, seed: int = 0) -> List[str]:
        if self._ids is None or len(self._ids) == 0:
            return []
        rng = random.Random(seed)
        return [str(rng.choice(self._ids)) for _ in range(n)]

    def all_chunk_ids(self) -> List[str]:
        if self._ids is None:
            return []
        return [str(c) for c in self._ids.tolist()]


# ---------------------------------------------------------------------------
# LLM wrapper (llama-cpp-python; fallback to transformers; final fallback: skip)
# ---------------------------------------------------------------------------


class _LLMBackend:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._llama = None
        self._hf = None
        if cfg.skip_llm:
            return
        if self._try_llama_cpp():
            return
        if self._try_transformers():
            return
        print("[rag] no LLM backend available — running retrieval-only",
              file=sys.stderr)

    def _try_llama_cpp(self) -> bool:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover
            print(f"[rag] llama-cpp-python unavailable ({exc})", file=sys.stderr)
            return False
        if not Path(self.cfg.model_path).exists():
            print(f"[rag] model file not found: {self.cfg.model_path}",
                  file=sys.stderr)
            return False
        try:
            self._llama = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_gpu_layers=self.cfg.n_gpu_layers,
                n_threads=self.cfg.n_threads,
                seed=self.cfg.seed,
                verbose=False,
            )
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[rag] llama-cpp init failed ({exc})", file=sys.stderr)
            return False

    def _try_transformers(self) -> bool:  # pragma: no cover - heavy dep path
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception:
            return False
        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
            mdl = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-1.5B-Instruct",
                load_in_4bit=True,
                device_map="auto",
            )
            self._hf = (tok, mdl)
            return True
        except Exception as exc:
            print(f"[rag] transformers fallback failed ({exc})", file=sys.stderr)
            return False

    def generate(self, prompt: str) -> Optional[str]:
        if self._llama is not None:
            out = self._llama(
                prompt,
                max_tokens=self.cfg.max_tokens,
                temperature=0.2,
                top_p=0.9,
            )
            try:
                return out["choices"][0]["text"]
            except Exception:
                return str(out)
        if self._hf is not None:  # pragma: no cover
            tok, mdl = self._hf
            inputs = tok(prompt, return_tensors="pt").to(mdl.device)
            out = mdl.generate(**inputs, max_new_tokens=self.cfg.max_tokens)
            return tok.decode(out[0], skip_special_tokens=True)
        return None


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------


class RAGQueryEngine:
    """End-to-end RAG pipeline plumbed through the tiered storage simulator."""

    def __init__(
        self,
        sim: TieredStorageSimulator,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.sim = sim
        self.cfg = config or EngineConfig()
        self._embedder = _QueryEmbedder(self.cfg.embedding_model)
        self._retriever = _Retriever(self.cfg.ssd_index_root)
        self._llm = _LLMBackend(self.cfg)
        self._retriever_ready = self._retriever._ids is not None
        if not self._retriever_ready:
            print(
                "[rag] Retriever has no index — all queries will return 0 "
                "chunks.\n"
                "[rag] Build the corpus first (see docs/USAGE.md §1).",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    def known_chunk_ids(self) -> List[str]:
        return self._retriever.all_chunk_ids()

    def random_chunk_ids(self, n: int, seed: int = 0) -> List[str]:
        return self._retriever.random_chunk_ids(n, seed=seed)

    # ------------------------------------------------------------------

    def query(self, q: str, *, top_k: Optional[int] = None) -> QueryResult:
        """Run one full retrieval (+ optional generation) cycle."""
        t0 = time.perf_counter()
        result = QueryResult(query=q)
        k = top_k or self.cfg.top_k

        # 1. Embed (encode() is guaranteed not to raise)
        qvec = self._embedder.encode(q) if self._retriever_ready else None

        # 2. Retrieve
        contexts: List[str] = []
        if qvec is not None:
            hits = self._retriever.search(qvec, k)
            for cid, score in hits:
                pre_hits = self.sim._cache.hits
                pre_misses = self.sim._cache.misses
                data = self.sim.read(cid)
                # Distinguish cache hit/miss by deltas in cache stats.
                if self.sim._cache.hits > pre_hits:
                    result.cache_hits += 1
                else:
                    result.cache_misses += 1
                if data is None:
                    continue
                result.retrieved_chunks.append(cid)
                result.retrieved_scores.append(score)
                result.bytes_read += len(data)
                # Decode text-only chunks; images stay as bytes-only context.
                try:
                    text = data.decode("utf-8", errors="ignore")[:1024]
                    contexts.append(text)
                except Exception:
                    pass

        # 3. Generate (optional)
        if not self.cfg.skip_llm and contexts:
            ctx = "\n---\n".join(contexts)
            prompt = (
                "You are a Wikipedia QA assistant.\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {q}\nAnswer:"
            )
            answer = self._llm.generate(prompt)
            result.answer = answer

        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------

    def read_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Force-read a list of chunk_ids (no LLM, no embedding).

        Used by Scenario B (cold sequential scan).  Returns total bytes read.
        """
        total = 0
        for cid in chunk_ids:
            data = self.sim.read(cid)
            if data is not None:
                total += len(data)
        return total

    # ------------------------------------------------------------------

    def ingest(self, chunk_id: str, payload: bytes,
               *, kind: str = "text") -> None:
        """Append a new doc to the corpus.  Used by Scenario C."""
        self.sim.write(chunk_id, payload, kind=kind)
