#!/usr/bin/env python3
"""Serve the playable Y-topology Beer Distribution Game.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."   # required for LLM opponents
  pip3 install -e ".[web,marl]"
  python3 scripts/serve_game.py
  # open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    """Load repo-root .env into os.environ if present (never committed)."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Playable Beer Distribution Game")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ippo-checkpoint",
        type=Path,
        default=None,
        help="Optional path to IPPO checkpoints/ directory",
    )
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            'Missing web deps. Install with: pip3 install -e ".[web]"',
            file=sys.stderr,
        )
        sys.exit(1)

    from beer_distribution_rl.agents.llm.openrouter import openrouter_api_key
    from beer_distribution_rl.web.runner import GameRunner
    from beer_distribution_rl.web.server import create_app

    if openrouter_api_key() is None:
        print(
            "Warning: OPENROUTER_API_KEY is not set. LLM opponents will fail until "
            "you export it or add it to a local .env file.",
            file=sys.stderr,
        )

    runner = GameRunner(seed=args.seed, ippo_checkpoint_dir=args.ippo_checkpoint)
    app = create_app(runner)

    print(f"Beer Distribution Game → http://{args.host}:{args.port}")
    if args.reload:
        uvicorn.run(
            "beer_distribution_rl.web.server:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
