#!/usr/bin/env bash
# ==============================================================================
# clear_caches.sh - Clear OS page cache, HDD write cache, and IMRSim counters
#
# Run this before each experiment to ensure a clean, comparable baseline:
#   - Drop Linux VFS/page/dentry/inode caches
#   - Flush the HDD's internal write cache (replace /dev/sdd with your device)
#   - Reset IMRSim per-zone RMW counters
#
# Requires: sudo privileges
#
# Usage:
#   sudo bash scripts/clear_caches.sh
#   sudo bash scripts/clear_caches.sh --device /dev/mapper/imrsim \
#       --hdd /dev/sdd \
#       --imrsim-util ~/IMRSim/imrsim_util/imrsim_util
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HDD_DEV="${HDD_DEV:-/dev/sdd}"
IMRSIM_DEV="${IMRSIM_DEV:-/dev/mapper/imrsim}"
IMRSIM_UTIL="${IMRSIM_UTIL:-${HOME}/IMRSim/imrsim_util/imrsim_util}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hdd)          HDD_DEV="$2";      shift ;;
        --device)       IMRSIM_DEV="$2";   shift ;;
        --imrsim-util)  IMRSIM_UTIL="$2";  shift ;;
        --help)
            echo "Usage: sudo $0 [--hdd /dev/sdX] [--device /dev/mapper/imrsim] [--imrsim-util PATH]"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[clear_caches] $*"; }

# ---------------------------------------------------------------------------
# 1. Flush dirty pages to disk
# ---------------------------------------------------------------------------
log "sync: flushing dirty pages..."
sync

# ---------------------------------------------------------------------------
# 2. Drop Linux page cache, dentries, and inodes
# ---------------------------------------------------------------------------
log "Dropping page/dentry/inode caches (echo 3 > /proc/sys/vm/drop_caches)..."
echo 3 | tee /proc/sys/vm/drop_caches > /dev/null

# ---------------------------------------------------------------------------
# 3. Flush HDD write cache (if device exists)
# ---------------------------------------------------------------------------
if [ -b "$HDD_DEV" ]; then
    log "Flushing HDD write cache on $HDD_DEV (hdparm -F)..."
    hdparm -F "$HDD_DEV" 2>/dev/null || log "WARNING: hdparm -F failed (non-fatal)"
else
    log "HDD device $HDD_DEV not found — skipping hdparm flush."
fi

# ---------------------------------------------------------------------------
# 4. Reset IMRSim per-zone RMW counters (s 4)
# ---------------------------------------------------------------------------
if [ -x "$IMRSIM_UTIL" ] && [ -b "$IMRSIM_DEV" ]; then
    log "Resetting IMRSim counters: $IMRSIM_UTIL $IMRSIM_DEV s 4"
    "$IMRSIM_UTIL" "$IMRSIM_DEV" s 4
else
    if [ ! -x "$IMRSIM_UTIL" ]; then
        log "WARNING: imrsim_util not found or not executable at $IMRSIM_UTIL"
    fi
    if [ ! -b "$IMRSIM_DEV" ]; then
        log "WARNING: IMRSim device $IMRSIM_DEV not found"
    fi
    log "Skipping IMRSim counter reset."
fi

# ---------------------------------------------------------------------------
# 5. Short settle time
# ---------------------------------------------------------------------------
sleep 1

log "Done. Caches cleared."
