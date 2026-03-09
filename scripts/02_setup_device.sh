#!/usr/bin/env bash
# ==============================================================================
# 02_setup_device.sh - Create a loop device + dmsetup target for IMRSim
#
# Creates:
#   /dev/loop0    <- backed by LOOP_FILE (a sparse image)
#   /dev/mapper/imrsim  <- device-mapper target using the IMRSim dm target
#   /mnt/imrsim/  <- ext4 mount point for experiments
#
# Requires:
#   - imrsim kernel module already loaded (run 01_install_imrsim.sh first)
#   - Root / sudo privileges
#   - losetup, dmsetup, mkfs.ext4, mount
#
# Usage:
#   sudo bash scripts/02_setup_device.sh
#   bash scripts/02_setup_device.sh --dry-run
#   bash scripts/02_setup_device.sh --size-gb 20 --mount /mnt/imrsim
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LOOP_FILE="${LOOP_FILE:-/var/tmp/imrsim_disk.img}"
DEVICE_NAME="${DEVICE_NAME:-imrsim}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/imrsim}"
SIZE_GB="${SIZE_GB:-10}"
DRY_RUN=false
FORMAT=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true ;;
        --format)     FORMAT=true ;;
        --size-gb)    SIZE_GB="$2"; shift ;;
        --loop-file)  LOOP_FILE="$2"; shift ;;
        --mount)      MOUNT_POINT="$2"; shift ;;
        --help)
            echo "Usage: $0 [--dry-run] [--format] [--size-gb N] [--loop-file PATH] [--mount PATH]"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        echo "[RUN] $*"
        "$@"
    fi
}

log() { echo "[02_setup_device] $*"; }

require_root() {
    if [ "$DRY_RUN" = false ] && [ "$(id -u)" -ne 0 ]; then
        echo "Error: this script must be run as root (use sudo)." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "Setting up IMRSim block device"
log "  Loop file  : $LOOP_FILE (${SIZE_GB} GB sparse)"
log "  DM target  : /dev/mapper/$DEVICE_NAME"
log "  Mount point: $MOUNT_POINT"
log "  DRY_RUN    : $DRY_RUN"

require_root

# 1. Verify kernel module is loaded
if [ "$DRY_RUN" = false ] && ! lsmod | grep -q imrsim; then
    echo "Error: imrsim kernel module is not loaded. Run 01_install_imrsim.sh first." >&2
    exit 1
fi

# 2. Create sparse backing image
if [ ! -f "$LOOP_FILE" ] || [ "$FORMAT" = true ]; then
    log "Creating ${SIZE_GB} GB sparse image at $LOOP_FILE..."
    run truncate -s "${SIZE_GB}G" "$LOOP_FILE"
fi

# 3. Attach loop device
log "Attaching loop device..."
LOOP_DEV=""
if [ "$DRY_RUN" = false ]; then
    # Detach first if already attached
    existing=$(losetup -j "$LOOP_FILE" 2>/dev/null | cut -d: -f1 | head -1)
    if [ -n "$existing" ]; then
        log "Loop device $existing already attached, reusing."
        LOOP_DEV="$existing"
    else
        LOOP_DEV=$(losetup --find --show "$LOOP_FILE")
        log "Attached $LOOP_DEV"
    fi
else
    LOOP_DEV="/dev/loop0"
    log "[DRY RUN] Would attach: losetup --find --show $LOOP_FILE -> $LOOP_DEV"
fi

# 4. Create device-mapper target
SECTORS=$(( SIZE_GB * 1024 * 1024 * 1024 / 512 ))
DM_TABLE="0 ${SECTORS} imrsim ${LOOP_DEV}"

if [ "$DRY_RUN" = false ] && dmsetup info "$DEVICE_NAME" &>/dev/null 2>&1; then
    log "Device /dev/mapper/$DEVICE_NAME already exists, removing..."
    run dmsetup remove "$DEVICE_NAME" || true
fi

log "Creating device-mapper target: $DM_TABLE"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] echo '$DM_TABLE' | dmsetup create $DEVICE_NAME"
else
    echo "$DM_TABLE" | dmsetup create "$DEVICE_NAME"
fi

# 5. Format (optional)
if [ "$FORMAT" = true ]; then
    log "Formatting /dev/mapper/$DEVICE_NAME as ext4..."
    run mkfs.ext4 -F "/dev/mapper/$DEVICE_NAME"
fi

# 6. Create mount point and mount
run mkdir -p "$MOUNT_POINT"
if [ "$DRY_RUN" = false ]; then
    if mountpoint -q "$MOUNT_POINT"; then
        log "$MOUNT_POINT already mounted."
    else
        log "Mounting /dev/mapper/$DEVICE_NAME -> $MOUNT_POINT"
        run mount "/dev/mapper/$DEVICE_NAME" "$MOUNT_POINT"
    fi
else
    echo "[DRY RUN] mount /dev/mapper/$DEVICE_NAME $MOUNT_POINT"
fi

log "Device setup complete."
log "  Block device: /dev/mapper/$DEVICE_NAME"
log "  Mount point : $MOUNT_POINT"

# Print usage reminder
cat <<EOF

Next steps:
  1. Generate the dataset:
       python workloads/synthetic_cv.py --root $MOUNT_POINT/imagenet
  2. Run the baseline experiment:
       python experiments/run_baseline.py --data-root $MOUNT_POINT/imagenet
  3. Run IMR-Fit:
       python experiments/run_imrfit.py --data-root $MOUNT_POINT/imagenet
  4. Plot results:
       python experiments/plot_results.py
EOF
