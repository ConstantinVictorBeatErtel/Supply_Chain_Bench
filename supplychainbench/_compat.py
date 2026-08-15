"""Import-path compatibility for the repository's bundled Hub environment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "beer_distribution_game"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))
