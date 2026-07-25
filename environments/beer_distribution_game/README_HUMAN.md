# Human baseline Gradio app

Anonymous human play on the **Tier 5 Y** beer game as **Retailer B** (36 weeks, orders 0–128, Hub seeds from development + validation). Counterparties are the same scripted policies used in Hub eval; observations stay fog-of-war identical to the model prompt.

## Local run

```bash
cd environments/beer_distribution_game
python3 -m pip install -r requirements-space.txt
PYTHONPATH=. python3 human_app/app.py
```

Optional logging to a Hugging Face Dataset:

```bash
export HF_TOKEN=hf_...                    # write-capable token (Space secret name)
export HF_DATASET_REPO=you/beer-game-human-sessions  # optional override
PYTHONPATH=. python3 human_app/app.py
```

If `HF_TOKEN` is unset or Hub sync fails, sessions append to local `human_sessions.jsonl` and gameplay continues.

## Space secrets / env

| Name | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | for Hub sync | Write token; Space secret |
| `HF_DATASET_REPO` | no | Dataset repo id; default `{whoami}/beer-game-human-sessions` |

`CommitScheduler` appends `sessions.jsonl` every **5 minutes** to that dataset repo (public). Logging errors never interrupt play.

## Data collected

Per finished or abandoned session: session UUID, timestamp, env version, tier, role, seed, actions, weekly inventory/backlog/cost, final cost / base-stock cost / reward when completed, prior beer-game experience (`yes`/`no`/`unsure`). No names, emails, IPs, or free text.

## App entry

- Module: `human_app/app.py`
- Gradio `demo` object is exported for Spaces (`demo = build_demo()`).
