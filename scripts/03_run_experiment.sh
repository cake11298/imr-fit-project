#!/usr/bin/env bash
# ==============================================================================
# 03_run_experiment.sh - One-shot experiment runner
#
# Runs both baseline and IMR-Fit experiments back-to-back, then generates
# all result plots.  Designed to be invoked inside the VM after the device
# has been set up with 02_setup_device.sh.
#
# Usage:
#   bash scripts/03_run_experiment.sh
#   bash scripts/03_run_experiment.sh --dry-run --epochs 5
#   bash scripts/03_run_experiment.sh --epochs 20 --budget 8 --theta 0.6
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/imrsim/imagenet}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CLASSES="${CLASSES:-10}"
IMAGES_PER_CLASS="${IMAGES_PER_CLASS:-100}"
BUDGET="${BUDGET:-10}"
THETA="${THETA:-0.55}"
W_FREQ="${W_FREQ:-0.35}"
W_SEQ="${W_SEQ:-0.30}"
W_SIZE="${W_SIZE:-0.15}"
W_REC="${W_REC:-0.20}"
DRY_RUN=false
SKIP_BASELINE=false
SKIP_IMRFIT=false
SKIP_PLOT=false
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=true ;;
        --epochs)             EPOCHS="$2"; shift ;;
        --batch-size)         BATCH_SIZE="$2"; shift ;;
        --data-root)          DATA_ROOT="$2"; shift ;;
        --classes)            CLASSES="$2"; shift ;;
        --images-per-class)   IMAGES_PER_CLASS="$2"; shift ;;
        --budget)             BUDGET="$2"; shift ;;
        --theta)              THETA="$2"; shift ;;
        --skip-baseline)      SKIP_BASELINE=true ;;
        --skip-imrfit)        SKIP_IMRFIT=true ;;
        --skip-plot)          SKIP_PLOT=true ;;
        --python)             PYTHON="$2"; shift ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --dry-run                 No real device I/O"
            echo "  --epochs N                Training epochs (default: 10)"
            echo "  --batch-size N            Batch size (default: 64)"
            echo "  --data-root PATH          Dataset root (default: /mnt/imrsim/imagenet)"
            echo "  --classes N               Number of classes to generate (default: 10)"
            echo "  --images-per-class N      Images per class (default: 100)"
            echo "  --budget N                Migration budget per epoch (default: 10)"
            echo "  --theta FLOAT             Placement threshold (default: 0.55)"
            echo "  --skip-baseline           Skip baseline experiment"
            echo "  --skip-imrfit             Skip IMR-Fit experiment"
            echo "  --skip-plot               Skip plot generation"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[03_run_experiment] $*"; }
hr()  { echo "======================================================================"; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DRY_FLAG=""
[ "$DRY_RUN" = true ] && DRY_FLAG="--dry-run"

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
hr
log "IMR-Fit Experiment Runner"
log "Timestamp  : $TIMESTAMP"
log "Repo root  : $REPO_ROOT"
log "Data root  : $DATA_ROOT"
log "Epochs     : $EPOCHS"
log "Dry run    : $DRY_RUN"
hr

cd "$REPO_ROOT"

# Check Python
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: Python not found at '$PYTHON'. Set PYTHON= env var." >&2
    exit 1
fi
log "Python: $($PYTHON --version)"

# Install dependencies if needed
if ! "$PYTHON" -c "import imrfit" &>/dev/null 2>&1; then
    log "Installing Python dependencies..."
    "$PYTHON" -m pip install -q -r requirements.txt || true
fi

# ---------------------------------------------------------------------------
# Step 1: Generate dataset (if not already present and not dry-run)
# ---------------------------------------------------------------------------
hr
log "Step 1: Dataset preparation"
if [ "$DRY_RUN" = false ] && [ ! -d "$DATA_ROOT" ]; then
    log "Generating synthetic CV dataset at $DATA_ROOT..."
    "$PYTHON" workloads/synthetic_cv.py \
        --root "$DATA_ROOT" \
        --classes "$CLASSES" \
        --images-per-class "$IMAGES_PER_CLASS"
else
    log "Dataset exists or dry-run; skipping generation."
fi

# ---------------------------------------------------------------------------
# Step 2: Baseline experiment
# ---------------------------------------------------------------------------
if [ "$SKIP_BASELINE" = false ]; then
    hr
    log "Step 2: Baseline experiment (naive IMR, no optimisation)"
    "$PYTHON" experiments/run_baseline.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-root "$DATA_ROOT" \
        --classes "$CLASSES" \
        --images-per-class "$IMAGES_PER_CLASS" \
        $DRY_FLAG
    log "Baseline complete."
else
    log "Step 2: Skipping baseline."
fi

# ---------------------------------------------------------------------------
# Step 3: IMR-Fit experiment
# ---------------------------------------------------------------------------
if [ "$SKIP_IMRFIT" = false ]; then
    hr
    log "Step 3: IMR-Fit experiment"
    "$PYTHON" experiments/run_imrfit.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-root "$DATA_ROOT" \
        --classes "$CLASSES" \
        --images-per-class "$IMAGES_PER_CLASS" \
        --budget "$BUDGET" \
        --theta "$THETA" \
        --w-freq "$W_FREQ" \
        --w-seq "$W_SEQ" \
        --w-size "$W_SIZE" \
        --w-rec "$W_REC" \
        $DRY_FLAG
    log "IMR-Fit complete."
else
    log "Step 3: Skipping IMR-Fit."
fi

# ---------------------------------------------------------------------------
# Step 4: Plot results
# ---------------------------------------------------------------------------
if [ "$SKIP_PLOT" = false ]; then
    hr
    log "Step 4: Generating figures"
    "$PYTHON" experiments/plot_results.py
    log "Figures saved to results/figures/"
else
    log "Step 4: Skipping plot."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
hr
log "Experiment run complete."
log "Results directory:"
ls -lh "$REPO_ROOT/results/" 2>/dev/null || echo "  (empty)"
if [ -d "$REPO_ROOT/results/figures" ]; then
    log "Figures:"
    ls -lh "$REPO_ROOT/results/figures/" 2>/dev/null
fi
hr
