"""Plot longitudinal RESET/MEMORY/LEARN result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--learn", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("continual-learning.png"))
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install matplotlib with `pip install -e '.[benchmark]'`") from exc
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, path, color in (("RESET", args.reset, "#6b7280"), ("MEMORY", args.memory, "#0b5d6b"), ("LEARN", args.learn, "#a65d21")):
        payload = json.loads(path.read_text())
        rows = payload.get("adaptation", [])
        values = [row.get("normalized_score") for row in rows]
        values = [float(value) if value is not None else float("nan") for value in values]
        ax.plot(range(1, len(values) + 1), values, label=label, color=color, alpha=0.85)
    ax.set_xlabel("Adaptation episode")
    ax.set_ylabel("Episode normalized score")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
