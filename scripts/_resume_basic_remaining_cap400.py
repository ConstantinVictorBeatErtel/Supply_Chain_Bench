#!/usr/bin/env python3
"""Resume only remaining burst_and_collapse seeds for untrained capacity-400 eval.

Named distinctly so sibling pkill -f eval_live_y* does not abort this job.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_live_y_domain_randomized_grpo_v1_models.py"

sys.argv = [
    str(SCRIPT),
    "--label",
    "untrained_qwen_capacity_400",
    "--model-name",
    str(ROOT / "local_checkpoints" / "qwen35-4b-base"),
    "--seeds",
    "b7c3c42c06605a25,3385fc1772a83c0f,6004dde3fcee7f67,f0cd8b5418cd1e55",
    "--output",
    str(ROOT / "artifacts" / "live_y_capacity_400" / "evaluations" / "untrained_qwen_capacity_400.json"),
]
runpy.run_path(str(SCRIPT), run_name="__main__")
