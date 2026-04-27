"""run_experiment.py — IMR-Fit end-to-end experiment driver.

Orchestrates Modules 1-5 in order:

    [1] Corpus build          (corpus.build_corpus)
    [2] Workload generation   (rag.scenarios via storage.tier_simulator)
    [3] Trace analysis        (imrfit.analyzer)
    [4] Trace replay          (imrsim.replay)
    [5] Aggregate results     (results/run_<timestamp>/summary.json)

CLI examples
------------
    # Quick smoke test (no LLM, 4 GB synthetic corpus, scenario A only)
    python run_experiment.py --scenario a --skip-llm --subset 0.2 --synthetic

    # Full experiment, three scenarios, real Wikipedia
    python run_experiment.py --scenario all

    # Analyse + replay only (corpus + traces already on disk)
    python run_experiment.py --analyze-only --fallback-imrsim

Outputs land in results/run_<UTC>/ and a `latest` symlink is updated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the project importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from corpus.build_corpus import (
    CorpusBuilder, CorpusConfig,
    DEFAULT_HDD_ROOT, DEFAULT_SSD_ROOT,
)
from corpus.verify_corpus import verify as verify_corpus
from storage.tier_simulator import TierConfig
from rag.scenarios import ScenarioRunner
from rag.query_engine import EngineConfig
from imrfit.analyzer import (
    Analyzer, AnalyzerConfig, grid_search_weights,
)
from imrfit.scorer import ScorerConfig
from imrsim.replay import Replayer, ReplayConfig


# ---------------------------------------------------------------------------
# Run directory bookkeeping
# ---------------------------------------------------------------------------


def _new_run_dir(root: str = "results") -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run = Path(root) / f"run_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    # Update the "latest" pointer (best-effort; symlink may fail on Windows).
    latest = Path(root) / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink():
                latest.unlink()
            elif latest.is_dir():
                shutil.rmtree(latest)
        latest.symlink_to(run.name)
    except OSError:
        pass
    return run


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_corpus(args, run_dir: Path) -> Dict[str, Any]:
    if args.skip_corpus:
        return {"skipped": True}
    target_gb = args.target_gb * args.subset
    cfg = CorpusConfig(
        hdd_root=args.hdd_root,
        ssd_root=args.ssd_root,
        target_size_gb=target_gb,
        use_synthetic=args.synthetic,
    )
    builder = CorpusBuilder(cfg)
    summary = builder.build()
    _save_json(summary, run_dir / "01_corpus.json")
    verify = verify_corpus(args.hdd_root, args.ssd_root)
    _save_json(verify, run_dir / "01_corpus_verify.json")
    return {"build": summary, "verify": verify}


def step_workload(args, run_dir: Path) -> Dict[str, Any]:
    scenarios = _resolve_scenarios(args.scenario)
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    eng_cfg = EngineConfig(
        ssd_index_root=str(Path(args.ssd_root)),
        skip_llm=args.skip_llm,
    )
    tier_cfg = TierConfig(
        hdd_root=args.hdd_root,
        ssd_root=args.ssd_root,
        cache_fraction=args.cache_fraction,
        block_size=args.block_size,
    )
    runner = ScenarioRunner(
        trace_dir=str(trace_dir),
        queries_per_scenario=args.queries,
        engine_config=eng_cfg,
        tier_config=tier_cfg,
    )
    reports = runner.run(scenarios)
    payload = {name: r.to_dict() for name, r in reports.items()}
    _save_json(payload, run_dir / "02_scenarios.json")
    return payload


def step_analyze(args, run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    trace_dir = Path(args.trace_dir)
    cfg = AnalyzerConfig(
        block_size=args.block_size,
        scorer_config=ScorerConfig(
            w_freq=args.w_freq, w_seq=args.w_seq,
            w_size=args.w_size, w_rec=args.w_rec,
            theta=args.theta,
        ),
    )
    analyzer = Analyzer(cfg)
    for trace in sorted(trace_dir.glob("scenario_*.jsonl")):
        report = analyzer.analyze(str(trace))
        slug = trace.stem
        analyzer.write_decisions(report,
                                 str(run_dir / f"03_{slug}_placement.jsonl"))
        analyzer.write_summary(report,
                               str(run_dir / f"03_{slug}_summary.json"))
        # Keep the top-level summary slim (no per-block dump).
        out[slug] = {
            "summary": report["summary"],
            "score_variance": report["score_variance"],
            "distributions": report["distributions"],
        }

        if args.grid_search:
            gs = grid_search_weights(str(trace), theta=args.theta)
            _save_json(gs, run_dir / f"03_{slug}_grid_search.json")
            out[slug]["grid_search_top10"] = gs[:10]

    _save_json(out, run_dir / "03_analyze.json")
    return out


def step_replay(args, run_dir: Path) -> Dict[str, Any]:
    trace_dir = Path(args.trace_dir)
    out: Dict[str, Any] = {}
    cfg = ReplayConfig(
        backend="python" if args.fallback_imrsim else args.replay_backend,
        block_size=args.block_size,
        epoch_io_count=args.epoch_io,
        migration_budget=args.budget,
        theta=args.theta,
        scorer_config=ScorerConfig(
            w_freq=args.w_freq, w_seq=args.w_seq,
            w_size=args.w_size, w_rec=args.w_rec,
            theta=args.theta,
        ),
    )
    replayer = Replayer(cfg)

    for trace in sorted(trace_dir.glob("scenario_*.jsonl")):
        slug = trace.stem
        scenario_label = slug.split("_")[-1].upper()
        results = replayer.replay(str(trace), scenario=scenario_label)
        out[slug] = {k: v.to_dict() for k, v in results.items()}
        _save_json(out[slug], run_dir / f"04_{slug}_replay.json")

    _save_json(out, run_dir / "04_replay.json")
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_scenarios(arg: str) -> List[str]:
    if arg == "all":
        return ["a", "b", "c"]
    return [s.strip().lower() for s in arg.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_dir = _new_run_dir(args.results_root)
    print(f"[run] results -> {run_dir}", file=sys.stderr)

    summary: Dict[str, Any] = {"args": vars(args), "run_dir": str(run_dir)}

    if not args.analyze_only:
        summary["corpus"] = step_corpus(args, run_dir)
        summary["workload"] = step_workload(args, run_dir)
    else:
        summary["corpus"] = {"skipped": "analyze-only"}
        summary["workload"] = {"skipped": "analyze-only"}

    summary["analyze"] = step_analyze(args, run_dir)
    summary["replay"] = step_replay(args, run_dir)

    _save_json(summary, run_dir / "summary.json")
    print(f"[run] done; summary at {run_dir / 'summary.json'}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Scenario / mode
    p.add_argument("--scenario", default="all",
                   help="Comma-separated subset, or 'all' (default)")
    p.add_argument("--queries", type=int, default=300,
                   help="Queries per scenario (default 300)")
    p.add_argument("--skip-corpus", action="store_true",
                   help="Reuse existing corpus on /mnt/hdd")
    p.add_argument("--skip-llm", action="store_true",
                   help="Don't call the LLM (retrieval-only workload)")
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip corpus + workload; analyse pre-existing traces")
    p.add_argument("--fallback-imrsim", action="store_true",
                   help="Force the Python IMR fallback even if /dev/mapper "
                        "/imrsim exists")
    p.add_argument("--replay-backend", choices=["python", "kernel"],
                   default="python")
    p.add_argument("--grid-search", action="store_true",
                   help="Run weight grid-search per scenario")

    # Paths
    p.add_argument("--hdd-root", default=DEFAULT_HDD_ROOT)
    p.add_argument("--ssd-root", default=DEFAULT_SSD_ROOT)
    p.add_argument("--trace-dir", default="traces")
    p.add_argument("--results-root", default="results")

    # Corpus
    p.add_argument("--target-gb", type=float, default=20.0,
                   help="Full-corpus target size (multiplied by --subset)")
    p.add_argument("--subset", type=float, default=1.0,
                   help="Fraction of --target-gb to actually build "
                        "(e.g. 0.2 for a 4 GB sandbox)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use deterministic synthetic corpus (no HF download)")

    # Storage tier
    p.add_argument("--cache-fraction", type=float, default=0.15)
    p.add_argument("--block-size", type=int, default=128 * 1024 * 1024)

    # IMR-Fit weights
    p.add_argument("--w-freq", type=float, default=0.35)
    p.add_argument("--w-seq", type=float, default=0.30)
    p.add_argument("--w-size", type=float, default=0.15)
    p.add_argument("--w-rec", type=float, default=0.20)
    p.add_argument("--theta", type=float, default=0.55)

    # Replay
    p.add_argument("--epoch-io", type=int, default=5000)
    p.add_argument("--budget", type=int, default=32,
                   help="Migration budget per epoch (IMR-Fit only)")
    return p


if __name__ == "__main__":
    sys.exit(main())
