#!/usr/bin/env bash
# ==============================================================================
# 01_install_imrsim.sh - Build and install the IMRSim kernel module
#
# Requires:
#   - Linux kernel headers matching the running kernel
#   - build-essential / gcc / make
#   - Root / sudo privileges
#   - IMRSim source code (cloned from the IMRSim repo)
#
# Usage:
#   sudo bash scripts/01_install_imrsim.sh
#   bash scripts/01_install_imrsim.sh --dry-run
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
IMRSIM_REPO="${IMRSIM_REPO:-https://github.com/example/imrsim.git}"  # update as needed
IMRSIM_DIR="${IMRSIM_DIR:-/opt/imrsim}"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help)
            echo "Usage: $0 [--dry-run]"
            echo "  --dry-run   Print commands without executing them"
            exit 0 ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1 ;;
    esac
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

log() { echo "[01_install_imrsim] $*"; }

require_root() {
    if [ "$DRY_RUN" = false ] && [ "$(id -u)" -ne 0 ]; then
        echo "Error: this script must be run as root (use sudo)." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "Starting IMRSim kernel module installation"
log "DRY_RUN=$DRY_RUN"

require_root

# 1. Install build dependencies
log "Installing kernel build dependencies..."
if command -v apt-get &>/dev/null; then
    run apt-get update -qq
    run apt-get install -y build-essential linux-headers-"$(uname -r)" git
elif command -v yum &>/dev/null; then
    run yum install -y gcc make kernel-devel-"$(uname -r)" git
else
    log "WARNING: Unknown package manager; ensure build-essential and kernel headers are installed."
fi

# 2. Clone IMRSim source
if [ ! -d "$IMRSIM_DIR" ]; then
    log "Cloning IMRSim source to $IMRSIM_DIR..."
    run git clone "$IMRSIM_REPO" "$IMRSIM_DIR"
else
    log "IMRSim directory already exists at $IMRSIM_DIR, pulling latest..."
    run git -C "$IMRSIM_DIR" pull --ff-only
fi

# 3. Build the kernel module
log "Building IMRSim kernel module..."
run make -C "$IMRSIM_DIR"

# 4. Install the module
log "Installing IMRSim kernel module..."
run insmod "$IMRSIM_DIR/imrsim.ko"

# 5. Verify
if [ "$DRY_RUN" = false ]; then
    if lsmod | grep -q imrsim; then
        log "SUCCESS: imrsim module is loaded."
        lsmod | grep imrsim
    else
        echo "ERROR: imrsim module failed to load." >&2
        dmesg | tail -20
        exit 1
    fi
else
    log "[DRY RUN] Would verify: lsmod | grep imrsim"
fi

log "IMRSim installation complete."
