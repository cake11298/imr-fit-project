"""RAG query engine driving the I/O workload.

The query engine binds three things together:
    * a tiered storage simulator (Module 2)
    * a FAISS index living on /mnt/ssd
    * an LLM (Qwen2-VL-2B Q4 via llama-cpp-python; or a no-LLM fallback)

It then runs three workload scenarios and emits a per-scenario JSONL
trace consumed by the analyzer (Module 4) and the IMRSim replayer
(Module 5).
"""

from .query_engine import (
    RAGQueryEngine,
    EngineConfig,
    QueryResult,
)
from .scenarios import (
    ScenarioRunner,
    SCENARIOS,
)

__all__ = [
    "RAGQueryEngine",
    "EngineConfig",
    "QueryResult",
    "ScenarioRunner",
    "SCENARIOS",
]
