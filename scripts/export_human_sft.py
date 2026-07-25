#!/usr/bin/env python3
"""Export completed human wholesaler sessions to chat SFT JSONL.

Replays each session on Hub BeerEpisode from (split, seed_index, actions) so
training examples use the identical observations the model sees at eval time.

Example:

    PYTHONPATH=environments/beer_distribution_game \\
      python3 scripts/export_human_sft.py \\
        --input human_sessions.jsonl \\
        --output data/human_demos/wholesaler_sft.jsonl \\
        --beat-base-stock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ENV = _ROOT / "environments" / "beer_distribution_game"
for path in (_ENV, _ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from human_app.replay import iter_sft_examples  # noqa: E402
from human_app.session import FIXED_ROLE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL of human session records (local fallback or HF dataset file)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination SFT JSONL (messages + meta per week)",
    )
    p.add_argument(
        "--action-format",
        choices=("json", "tool"),
        default="json",
        help="Assistant label format (json matches Colab GRPO pilot)",
    )
    p.add_argument(
        "--beat-base-stock",
        action="store_true",
        help="Keep only sessions with final_total_cost < base_stock_cost",
    )
    p.add_argument(
        "--role",
        default=FIXED_ROLE,
        help="Expected controlled role (default: wholesaler)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.role != FIXED_ROLE:
        raise SystemExit(
            f"export currently supports only {FIXED_ROLE!r} demos; got {args.role!r}"
        )
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_sessions = 0
    n_examples = 0
    n_skipped = 0
    with args.input.open(encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.input}:{line_no}: invalid JSON ({exc})") from exc
            if record.get("status") != "completed":
                n_skipped += 1
                continue
            if record.get("role", FIXED_ROLE) != FIXED_ROLE:
                n_skipped += 1
                continue
            wrote = 0
            try:
                for example in iter_sft_examples(
                    record,
                    action_format=args.action_format,
                    require_beat_base_stock=args.beat_base_stock,
                ):
                    dst.write(json.dumps(example, sort_keys=True) + "\n")
                    wrote += 1
            except Exception as exc:
                raise SystemExit(
                    f"{args.input}:{line_no}: replay failed ({exc})"
                ) from exc
            if wrote == 0:
                n_skipped += 1
            else:
                n_sessions += 1
                n_examples += wrote

    print(
        json.dumps(
            {
                "sessions_exported": n_sessions,
                "weekly_examples": n_examples,
                "sessions_skipped": n_skipped,
                "output": str(args.output),
                "role": FIXED_ROLE,
                "action_format": args.action_format,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
