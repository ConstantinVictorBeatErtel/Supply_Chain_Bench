# Supply Chain RL Environment & Benchmark (SupplyChainBench)

**[▶ Play the corrected stochastic v2 game](https://constantinvictorbeatertel.github.io/Supply_Chain_Bench/)**

**Can an AI agent learn to control a delayed, partially observed system?**

SupplyChainBench measures long-horizon decision-making under hidden and changing
dynamics through the beer distribution game. In the beer supply chain distribution game, an agent runs the wholesaler for 36 weeks.
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

The browser replay preserves the historical baseline, untrained-model, and
trained-model comparison. The playable game now runs the corrected stochastic
v2 training scenario: every new game gets a fresh seed, customer demand is
sampled each retailer/week, and retailer orders carry that demand variation to
the wholesaler. Until a model is retrained on v2, the end screen compares the
human only with an adaptive base-stock policy on the exact same episode seed.

The chain is a Y: one wholesaler splitting a single inventory pool between
two retailers who compete for it.

```mermaid
flowchart LR
  F[Factory] --> D[Distributor] --> WH["Wholesaler · your seat"]
  WH --> RA[Retailer A] --> CA[Customer A]
  WH --> RB[Retailer B] --> CB[Customer B]
  style WH fill:#7A3B45,color:#FFFFFF,stroke:#3A2F2C,stroke-width:2px
```

The playable wholesaler sees the combined orders placed by the two retailers,
not privileged end-customer demand. In v2, each retailer orders its newly sampled
customer demand plus the fixed scarcity increment. That preserves fog of war
without collapsing the wholesaler's incoming signal to a constant value.

The archived v1 model results remain reproducible and are not presented as v2
game comparisons. A new model comparison will require retraining and evaluation
on `live-y-domain-randomized-grpo-v2`.

## Technical benchmark details

36 weeks each, factory capacity 400. The LLM prompt
withholds the demand law and the capacity. 

![Live-Y capacity-400 scoreboard](docs/assets/live-y-capacity-400-benchmark-v4.png)

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

For every model that finished all 16 the numerator is the 287.22 above. The
latest fully clean API runs score 43.61 for GPT-5.6 Sol and 42.09 for Grok 4.6.
Runs with any protocol failure are diagnostic and remain unranked on the
canonical board: Claude Opus 5 completed 15/16 clean (diagnostic score 51.28),
the DeepSeek V4 Flash 0731 rerun completed 11/16 (7.28), and Muse Spark 1.2
completed 0/16 (no score). The previously published fully clean DeepSeek run
remains ranked at 7.12. Laguna (7/16 clean) and Nemotron (2/16) are likewise
small-sample diagnostics.

## Qwen fine-tune (live-Y GRPO v1)

**[▶ Explore the LoRA + GRPO learning loop](https://constantinvictorbeatertel.github.io/Supply_Chain_Bench/lora-grpo/)**

<p align="center">
  <a href="https://constantinvictorbeatertel.github.io/Supply_Chain_Bench/lora-grpo/">
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
