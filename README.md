# Beer Distribution Game

[▶ Play](https://beer-distribution-game.pages.dev/)

A stochastic supply chain game to teach the bullwhip effect. The human and/or LLM play one of two wholesalers. 
It is a y-topology so that the human can play directly against the LLM in one of the positions that experience the bullwhip 
effect. 

```mermaid
flowchart LR
  F[Factory] --> D[Distributor]
  D --> WH["Wholesaler · you"]
  D --> WM["Wholesaler · recorded model"]
  WH --> RA[Retailer A] --> CA[Customer A]
  WM --> RB[Retailer B] --> CB[Customer B]
  style WH fill:#7A3B45,color:#FFFFFF,stroke:#3A2F2C,stroke-width:2px
  style WM fill:#C9844A,color:#FFFFFF,stroke:#3A2F2C,stroke-width:2px
```

The playable board highlights two wholesalers: A human and the sealed recorded-model
companion on the same seed. Underneath, the Y DAG still has one wholesaler seat feeding Retailer A/B; the second card is
the parallel comparison episode, not a second live seat.

## Benchmark 

36 weeks. LLM prompt withholds demand law and capacity. 

![Live-Y capacity-400 scoreboard](docs/assets/live-y-capacity-400-benchmark-v3.png)

For reference on the same seeds: hindsight-perfect scores 100, an adaptive
base-stock heuristic 73.9, and a blind constant-order policy that never reads
the observation 19.8.

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
Across the 16 seeds, the reference means are:

$$
\overline{C^{\star}} = 287.22,
\qquad
\text{adaptive base-stock average} = 388.84
$$

Score on protocol-valid episodes:

$$
\mathrm{score} = 100 \times \frac{\overline{C^{\star}}}{\overline{C_{\mathrm{policy}}}}
$$

## Qwen fine-tune (live-Y GRPO v1)

[`scripts/train_live_y_domain_randomized_grpo_v1.py`](scripts/train_live_y_domain_randomized_grpo_v1.py)
trains a LoRA adapter on `Qwen/Qwen3.5-4B` with critic-free multi-turn
group-relative updates (GRPO-style). Research capacity is 400; Hub Tier-5
stays 22.

### What LoRA trains

The 4B base weights stay frozen in bf16. A rank-16 / alpha-16 LoRA adapter
(dropout 0) is attached to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`, and it is the only thing the optimizer touches. The
target behaviour — order roughly what you sold, and count the units already in
the pipeline — is a small behavioural adjustment rather than new knowledge, so
rank 16 across all seven projections is ample. It also keeps rollouts and
training on one GPU, since the frozen base is shared.

### What RL does with it

One update is: roll out, grade each decision, reweight the tokens that carried
it.

1. **Roll out.** Pick training seeds; play each one `group_size` times. Group
   members share a scenario, a demand seed, and counterparty streams (CRN), so
   they differ only through sampling. Each week the policy sees the role-local
   observation and emits `{"quantity": N}`.
2. **Grade.** Every decision gets a windowed return-to-go, group-normalized at
   the same timestep.
3. **Update.** A token-level clipped importance-ratio step on the LoRA
   parameters only.

Demand is `episode_randomized_y_poisson_v1`: per episode,
$\lambda \sim \mathcal{U}[2,8]$, then independent Poisson draws for each retailer.

**Credit window.** An order clears the order delay (1 week) and shipment delay
(2 weeks) within ~3 weeks and washes out by ~6, so a decision is charged for
$W = 6$ weeks of downstream local cost, not the whole remaining episode.
Settlement and terminal exposure are attributed only to decisions whose window
reaches the horizon $H$; a protocol failure inside the window contributes
$-10^{5}$:

$$
G_{i,t} = -\sum_{u=t}^{\min(t+W,\,H)-1} \gamma^{\,u-t} c_{i,u}
\;+\; \mathbf{1}[t + W \ge H]\;\gamma^{\,H-t}\,r^{\mathrm{term}}_{i}
$$

with $\gamma = 1$ by default. Unbounded return-to-go
($W \to \infty$, the archived setting) left only ~10% of the advantage variance
genuinely per-turn — see
[`docs/LIVE_Y_RL_POSTMORTEM.md`](docs/LIVE_Y_RL_POSTMORTEM.md).

**Advantage.** Same-timestep group mean and standard deviation, so the
surrogate sees a unit-scale signal rather than raw cost units:

$$
A_{i,t} = \frac{G_{i,t} - \mu_{g,t}}{\sigma_{g,t}},
\qquad
\mu_{g,t} = \frac{1}{|g|}\sum_{j \in g} G_{j,t}
$$

**Objective.** The ratio is per-token over the tokens that encode the order,
not a mean over the whole completion, and the clip is double-sided so a badly
off-policy token with a negative advantage cannot dominate a minibatch.
For each scored token $k$ of decision $(i,t)$, with
$r_k = \exp\big(\ell_\theta(y_k) - \ell_{\mathrm{old}}(y_k)\big)$ and
$s_k = \min\big(r_k A_{i,t},\; \mathrm{clip}(r_k, 1-\varepsilon, 1+\varepsilon)\,A_{i,t}\big)$:

$$
\mathcal{L} = -\,\mathbb{E}_{k}\big[\tilde{s}_k\big],
\qquad
\tilde{s}_k =
\begin{cases}
\max\big(s_k,\; c\,A_{i,t}\big) & A_{i,t} < 0\\[2pt]
s_k & A_{i,t} \ge 0
\end{cases}
$$

with $\varepsilon = 0.2$ and dual-clip $c = 3$. Only the digits of
`{"quantity": N}` are scored — averaging over the whole completion let seven
boilerplate tokens outvote the one token carrying the decision.

### Result

| | untrained | trained |
|---|---:|---:|
| capacity-400 score (16 held-out seeds) | 6.73 | **20.64** |
| mean local cost | 4264.7 | **1391.8** |
| protocol-clean episodes | 16/16 | **16/16** |

A 3.1× score improvement over the same base model, better on all four demand
buckets including the three never trained on. Mean weekly order fell from ~22.9
to ~17.5 against an obligation near 16/week — the over-ordering habit that
dominates cost in this game.

Two caveats worth stating up front: this clears a blind constant-order baseline
(19.80) by only 0.84 points, and both training runs used the same seed, so
run-to-run variance is unmeasured.

Full recipe in [`docs/TRAINING.md`](docs/TRAINING.md). An earlier two-update run
scored 7.48 — indistinguishable from the untrained base — and
[`docs/LIVE_Y_RL_POSTMORTEM.md`](docs/LIVE_Y_RL_POSTMORTEM.md) documents why it
could not have worked.

Weights: [`artifacts/live_y_best_adapter/`](artifacts/live_y_best_adapter/).

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

## Links

[Hugging Face Space](https://constantinertel-beer-distribution-game.static.hf.space/) · [Deploy](DEPLOY.md)

## Run locally

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
npm ci && npm run test:all && npm run build
```
