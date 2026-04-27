"""Corpus builder for the IMR-Fit experiment.

Produces a Wikipedia-derived multimodal corpus laid out on /mnt/hdd
(cold tier) with a FAISS index on /mnt/ssd (hot tier).
"""

from .build_corpus import (
    CorpusConfig,
    CorpusBuilder,
    DEFAULT_HDD_ROOT,
    DEFAULT_SSD_ROOT,
)

__all__ = [
    "CorpusConfig",
    "CorpusBuilder",
    "DEFAULT_HDD_ROOT",
    "DEFAULT_SSD_ROOT",
]
