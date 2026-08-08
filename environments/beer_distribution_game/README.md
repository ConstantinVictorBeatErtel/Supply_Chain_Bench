# Beer Distribution Game

A public Verifiers environment for evaluating tool-using LLMs on delayed
supply-chain control. One model controls one role through a strict
`place_order(quantity)` tool while deterministic scripted policies control the
other roles. The simulator is deterministic and replayable: the same scenario
seed and action trace always produce the same outcome.

This package is the **Prime Intellect Environments Hub / Verifiers**
deliverable. The frozen Tier-5 Y headline keeps **factory capacity 22**
(calibrated scarcity). The repository’s public browser game and live-Y research
board use **capacity 400** instead — see the root
[`README.md`](../../README.md) and
[`artifacts/live_y_capacity_400/`](../../artifacts/live_y_capacity_400/).

The environment provides five tiers: steady demand, persistent stochastic demand,
a hidden regime shift, partial pipeline observability, and strategic scarcity in a
two-retailer Y network. Episodes are seeded, exactly replayable, and graded from
the simulator trace. The primary result is controlled-role cost normalized against
a same-seed adaptive base-stock reference; service, bullwhip, and system externality
remain separate metrics.

## Human baseline Gradio app

Play Tier 5 Y as the **Wholesaler** (identical Hub seeds, FOW observations,
horizon, grading, and **Hub capacity 22**). See [`README_HUMAN.md`](README_HUMAN.md) and
[`docs/HUMAN_TO_MODEL.md`](../../docs/HUMAN_TO_MODEL.md).

For the public static game (capacity **400**, wholesaler-only UI), use
[`static_web/`](../../static_web/) / the Hugging Face Static Space — not this Gradio app.

```bash
python3 -m pip install -r requirements-space.txt
PYTHONPATH=. python3 human_app/app.py
```

## Local validation

From this directory with `uv` installed:

```bash
uv sync
uv run validate beer-distribution-game --runtime.type subprocess
```

Example dry-run configuration:

```bash
uv run eval @ eval.toml --dry-run True
```

The package exports one `BeerTaskset` and one bundled `BeerHarness`, following the
native Verifiers 0.2 contract. It requires no API keys for simulation or grading.
The checked-in evaluation config disables result upload by default.

## Current development evidence

The steady retailer task is a protocol/control screen. DeepSeek V4 Flash matched
the corrected base-stock reference exactly (cost 69, reward 0.5). The complementary
tiers 5 Y-wholesaler task is harder: across three development seeds the model cost
was 1,111.8 ± 213.2 versus 850.7 ± 326.1 for paired base-stock, with reward
0.423 ± 0.060. All 108 actions were protocol-clean. This is a development finding,
not a held-out or multi-model benchmark result.

Evaluation configurations are named by role and scope:

- `eval_akash_retailer_smoke.toml` and `eval_akash_retailer_dev.toml`;
- `eval_akash_wholesaler_y_smoke.toml` and
  `eval_akash_wholesaler_y_dev.toml`.

Start with a one-seed smoke. All Akash configurations run sequentially, keep
uploads disabled, and read the API key from `AKASH_API_KEY`; credentials are
never stored in configs or compact result artifacts.

See the repository-level specifications for the complete interface, reward, and
difficulty contracts.
