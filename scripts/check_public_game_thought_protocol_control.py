"""Control: same seed/capacity/model, published JSON-only protocol (no rationale).

Isolates whether asking for a per-week thought changes how well the model plays,
which is the one confound in `run_public_game_llm_thoughts.py`: those traces back
the public game's comparison opponent, so the rationale must not be buying or
costing performance.

Usage (from the repository root):
    OPENROUTER_API_KEY=sk-or-... \\
        python scripts/check_public_game_thought_protocol_control.py development 0
"""

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "beer_distribution_game"))
spec = importlib.util.spec_from_file_location(
    "runner", ROOT / "scripts" / "run_public_game_llm_thoughts.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import observation_user_message

PLAIN = (
    "Place exactly one order each week. Respond with exactly one JSON object of "
    'the form {"quantity": <integer from 0 through 128>}. Do not include any '
    "other text."
)


def plain_system(spec_):
    full = runner.system_prompt(spec_)
    return full.split("Place exactly one order each week.")[0] + PLAIN


def main() -> None:
    split, index = sys.argv[1], int(sys.argv[2])
    client = runner.OpenRouterClient(runner.DEFAULT_MODEL, budget_usd=0.6)
    scenario = runner.public_spec(split, index)
    episode = BeerEpisode(scenario, "wholesaler")
    observation = episode.start()
    actions, failures = [], 0
    while not episode.done:
        for attempt in range(3):
            raw = client.complete(
                system=plain_system(scenario),
                user=observation_user_message(observation),
                session_id=f"control|{split}|{index}",
            )
            quantity, _ = runner.parse_reply(raw)
            if quantity is not None:
                break
            failures += 1
        if quantity is None:
            raise RuntimeError(f"no order at week {observation['week']}")
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]["primary"]
    print(json.dumps({
        "seed": f"{split}-{index}",
        "protocol": "json-only control",
        "local_total_cost": grade["local_total_cost"],
        "base_stock": grade["paired_base_stock_local_total_cost"],
        "format_failures": failures,
        "actions": actions,
        "spent_usd": round(client.spent, 5),
    }))


if __name__ == "__main__":
    main()
