# SupplyChainBench: Beer Distribution

**[▶ Play the game](https://beer-distribution-game.pages.dev)**

**Can an AI agent learn to control a delayed, partially observed system—not
just produce a good one-shot answer?**

SupplyChainBench measures long-horizon decision-making under hidden and changing
dynamics. In Beer Distribution, an agent runs the wholesaler for 36 weeks.
Orders and shipments take time, mistakes are only visible weeks later, and the
agent is not told the demand law or supply limits.

The project measures three things:

- **Control:** can the agent keep inventory and backlog costs low?
- **Adaptation:** can it recover when demand, delays, or supply change?
- **Learning:** does improvement come from a bounded notebook or real LoRA
  weight updates across episodes?

## Benchmark

The main test uses 16 fixed supply chains. The agent controls the wholesaler
without being told the demand pattern or supply limit.

Five additional tests change demand, delivery times, or supply to see whether
the agent notices and recovers. Separate experiments compare starting fresh,
carrying short notes between games, and updating model weights. Their scores
stay separate so unlike tests are not compared directly.

The browser replay shows a simple baseline, the untrained model, and the trained
model facing the same supply chain.

The chain is a Y: one wholesaler splitting a single inventory pool between
two retailers who compete for it.

```mermaid
flowchart LR
  F[Factory] --> D[Distributor] --> WH["Wholesaler · your seat"]
  WH --> RA[Retailer A] --> CA[Customer A]
  WH --> RB[Retailer B] --> CB[Customer B]
  style WH fill:#7A3B45,color:#FFFFFF,stroke:#3A2F2C,stroke-width:2px
```

You and the recorded model each play that one wholesaler seat in your own sealed
episode, on the same seed against the same scripted counterparties — two runs of
the same chain, not two live seats in one chain. The playable board shows both
as a station pair so you can watch them diverge.

## Technical benchmark details

36 weeks each, factory capacity 400. The LLM prompt
withholds the demand law and the capacity. 

![Live-Y capacity-400 scoreboard](docs/assets/live-y-capacity-400-benchmark-v3.png)

For reference on the same seeds: hindsight-perfect scores 100, an adaptive
base-stock heuristic 73.9, and the best blind constant-order policy — order 18
every week, never reading the observation — 19.8.

Artifacts: [`artifacts/live_y_capacity_400/evaluations/`](artifacts/live_y_capacity_400/evaluations/).
The chart is generated from the leaderboard JSON by
[`scripts/render_capacity_400_scoreboard.py`](scripts/render_capacity_400_scoreboard.py),
so it cannot drift from the evaluations.

### Scoring

Weekly local cost (holding $0.5$, backlog $1.0$):

$$
c_t = 0.5\, I_t + 1.0\, B_t
$$

Episode cost over $H=36$ decision weeks, $3$ settlement weeks, and terminal
inventory-position charge ($O$ = on-order pipeline):

$$
\begin{aligned}
IP &= I + O - B \\
c^{\mathrm{term}} &= 0.5\max(IP,0) + 1.0\max(-IP,0) \\
C &= \sum_{t=1}^{H} c_t + \sum_{t=H+1}^{H+3} c_t + c^{\mathrm{term}}
\end{aligned}
$$

Hindsight-perfect $C^{\star}$ is a feasible open-loop upper bound on each CRN seed.
Across all 16 seeds the reference means are:

$$
\overline{C^{\star}} = 287.22,
\qquad
\text{adaptive base-stock average} = 388.84
$$

Both means are taken over the same protocol-clean subset $S$ of the 16 seeds, so
a model that fails episodes is scored against the perfect reference for the
seeds it actually finished:

$$
\mathrm{score} = 100 \times
\frac{\frac{1}{|S|}\sum_{s \in S} C^{\star}_{s}}
     {\frac{1}{|S|}\sum_{s \in S} C^{\mathrm{policy}}_{s}}
$$

For every model that finished all 16 the numerator is the 287.22 above; the two
free models did not, so their references are re-based (Laguna 7/16 clean,
$\overline{C^{\star}} = 362.36$; Nemotron 2/16, $352.75$). Read those two bars
as small-sample.

## Qwen fine-tune (live-Y GRPO v1)

**[▶ Explore the LoRA + GRPO learning loop](https://constantinvictorbeatertel.github.io/beer_distribution_RL/lora-grpo/)**

<p align="center">
  <a href="https://constantinvictorbeatertel.github.io/beer_distribution_RL/lora-grpo/">
    <img src="static_web/public/lora-grpo/og.png" width="720" alt="GRPO compares six matched rollouts and assigns delayed credit; the resulting optimizer step updates only the LoRA adapter while the Qwen base model remains frozen">
  </a>
</p>

*GRPO supplies the comparison and credit-assignment objective; LoRA is the
trainable parameterization that receives those gradients while the base model
stays frozen.*

[`scripts/train_live_y_domain_randomized_grpo_v1.py`](scripts/train_live_y_domain_randomized_grpo_v1.py)
trains a LoRA adapter on `Qwen/Qwen3.5-4B` with critic-free multi-turn
group-relative updates (GRPO-style). The Qwen base stays frozen; only a rank-16
LoRA adapter on the attention and MLP projections is optimized.

Each update rolls out 8 matched seeds 6 times, scores every order against its
groupmates using a 6-week downstream-cost window, and applies a clipped
token-level update to the digits in `{"quantity": N}`. In short, GRPO decides
which sampled actions were better; LoRA is the small set of weights changed to
make those actions more likely.

Full details: [`docs/TRAINING.md`](docs/TRAINING.md) ·
[`docs/LIVE_Y_RL_POSTMORTEM.md`](docs/LIVE_Y_RL_POSTMORTEM.md) ·
[`artifacts/live_y_best_adapter/`](artifacts/live_y_best_adapter/)

## Hyperefficient compute

Documented in [`docs/LIVE_Y_EFFICIENCY.md`](docs/LIVE_Y_EFFICIENCY.md):

- Prefix KV cache across shared chat-template tokens; invalidate after each LoRA update
- Batched rollouts; separate inference / train minibatches; LoRA + bf16 + gradient checkpointing
- Bounded JSON completions (32-token train cap); rolling history window, not full transcript
- `torch.inference_mode` for generation / old-policy scoring; critic-free group baselines (no value net)
- CRN / fixed seed derivation to cut rollout variance

## How it is built

| Layer | Role |
| --- | --- |
| Python oracle (`environments/…/core.py`, research env) | Seeded simulator + grader |
| [`static_web/`](static_web/) | Browser JS port, parity-checked against the oracle |
| [`beer_distribution_rl/`](beer_distribution_rl/) | Research protocol, agents, GRPO / eval harness |
| [`environments/beer_distribution_game/`](environments/beer_distribution_game/) | Verifiers Hub package (Tier-5 capacity 22) |

## Run locally

```bash
python -m pip install -e ".[dev,benchmark]"
python -m pytest -q
python -m supplychainbench.eval --model agent:constant-18 --suite standard
python -m supplychainbench.leaderboard
npm ci
npm run test:all
npm run build
```
