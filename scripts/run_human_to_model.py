#!/usr/bin/env python3
"""One-command human-demo → SFT → GRPO pipeline for Tier-5 Y wholesaler.

Where to run
------------
* **Export only** — any machine (Mac/CPU fine).
* **SFT / GRPO** — Google Colab (or any CUDA GPU). Local Macs without NVIDIA
  CUDA cannot train.

Default base model
------------------
``Qwen/Qwen3-0.6B`` — matches the existing Colab GRPO smoke and fits a Colab T4.
For the stronger training-gate target use::

    --model-name Qwen/Qwen2.5-7B-Instruct

Examples
--------
Validate export + dataset shape (CPU)::

    PYTHONPATH=environments/beer_distribution_game \\
      python3 scripts/run_human_to_model.py \\
        --sessions human_sessions.jsonl \\
        --dry-run

Full train on Colab GPU::

    PYTHONPATH=environments/beer_distribution_game \\
      python3 scripts/run_human_to_model.py \\
        --sessions human_sessions.jsonl \\
        --beat-base-stock
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ENV = _ROOT / "environments" / "beer_distribution_game"
_SCRIPTS = _ROOT / "scripts"

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SFT_DIR = _ROOT / "outputs" / "beer-wholesaler-sft"
DEFAULT_GRPO_DIR = _ROOT / "outputs" / "beer-wholesaler-grpo"
DEFAULT_SFT_JSONL = _ROOT / "data" / "human_demos" / "wholesaler_sft.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sessions",
        type=Path,
        required=True,
        help="Human sessions JSONL (from Gradio local fallback or HF dataset)",
    )
    p.add_argument(
        "--sft-jsonl",
        type=Path,
        default=DEFAULT_SFT_JSONL,
        help="Exported chat SFT JSONL path",
    )
    p.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    p.add_argument("--grpo-dir", type=Path, default=DEFAULT_GRPO_DIR)
    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--stages",
        default="export,sft,grpo",
        help="Comma list from: export,sft,grpo (default: all three)",
    )
    p.add_argument(
        "--beat-base-stock",
        action="store_true",
        help="Export only sessions that beat same-seed base-stock cost",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Export + validate SFT rows; skip GPU training",
    )
    p.add_argument("--grpo-updates", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260725)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()})}
    # Ensure Hub package + repo root import cleanly for child scripts.
    py_path = str(_ENV) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONPATH"] = py_path
    subprocess.run(cmd, check=True, cwd=str(_ROOT), env=env)


def main() -> int:
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    allowed = {"export", "sft", "grpo"}
    unknown = set(stages) - allowed
    if unknown:
        raise SystemExit(f"unknown stages: {sorted(unknown)}; choose from {sorted(allowed)}")
    if not stages:
        raise SystemExit("no stages selected")

    if not args.sessions.exists():
        raise SystemExit(
            f"sessions file not found: {args.sessions}\n"
            "Play the Gradio app first, or download sessions.jsonl from the HF dataset."
        )

    summary: dict = {
        "model_name": args.model_name,
        "stages": stages,
        "dry_run": args.dry_run,
        "where": (
            "CPU ok for export/dry-run; run sft+grpo on Google Colab (CUDA GPU)"
        ),
    }

    if "export" in stages:
        export_cmd = [
            sys.executable,
            str(_SCRIPTS / "export_human_sft.py"),
            "--input",
            str(args.sessions),
            "--output",
            str(args.sft_jsonl),
            "--action-format",
            "json",
        ]
        if args.beat_base_stock:
            export_cmd.append("--beat-base-stock")
        _run(export_cmd)
        summary["sft_jsonl"] = str(args.sft_jsonl)

    if "sft" in stages:
        sft_cmd = [
            sys.executable,
            str(_SCRIPTS / "train_colab_sft_wholesaler.py"),
            "--data",
            str(args.sft_jsonl),
            "--output-dir",
            str(args.sft_dir),
            "--model-name",
            args.model_name,
            "--seed",
            str(args.seed),
        ]
        if args.dry_run:
            sft_cmd.append("--dry-run")
        _run(sft_cmd)
        summary["sft_dir"] = str(args.sft_dir)
        summary["adapter"] = str(args.sft_dir / "adapter")

    if "grpo" in stages:
        if args.dry_run:
            print("dry-run: skipping GRPO GPU stage", flush=True)
        else:
            adapter = args.sft_dir / "adapter"
            grpo_cmd = [
                sys.executable,
                str(_SCRIPTS / "train_colab_grpo_wholesaler.py"),
                "--model-name",
                args.model_name,
                "--output-dir",
                str(args.grpo_dir),
                "--updates",
                str(args.grpo_updates),
                "--seed",
                str(args.seed),
            ]
            # Warm-start from SFT when that stage ran or adapter already exists.
            if adapter.exists() or "sft" in stages:
                grpo_cmd.extend(["--adapter", str(adapter)])
            _run(grpo_cmd)
            summary["grpo_dir"] = str(args.grpo_dir)

    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print(
            "\nNext (on Colab GPU): re-run the same command without --dry-run",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
