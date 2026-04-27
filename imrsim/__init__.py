"""IMRSim replay engine.

If the IMRSim kernel module is loaded and exposes /dev/mapper/imrsim, we
replay traces against the real device.  Otherwise the pure-Python
``FallbackIMRSim`` reproduces the Top/Bottom RMW penalty model in
software so the experiment can keep moving while IMRSim is being patched.
"""

from .replay import (
    Strategy,
    ReplayConfig,
    ReplayResult,
    Replayer,
)
from .fallback_simulator import FallbackIMRSim, RMWModel

__all__ = [
    "Strategy",
    "ReplayConfig",
    "ReplayResult",
    "Replayer",
    "FallbackIMRSim",
    "RMWModel",
]
