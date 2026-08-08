# Live-Y feasible-supply GRPO v2 (design; GPU launch gated)

**Status:** capacity-400 OpenRouter board published under
`artifacts/live_y_capacity_400/`. Do **not** launch a paid GPU retrain until
the user explicitly approves.

**Aborted pod:** `9k122r89nktor4` (`live-y-retrain-a40`) was stopped and
**removed** on 2026-08-08. Billing must not continue on that id.

## Root cause (why the aborted run looked starved)

Frozen Hub / Tier-5 Y (`scenario.py`) uses:

| Quantity | Value |
|---|---|
| Factory capacity | **22** |
| Customer mean demand | **7.5 × 2 retailers = 15** |
| Initial inventory (all roles) | **12** |
| Initial shipment / order pipelines | **4 / 4** |
| Rival retailers (headline) | `scarcity_aggressive_v1` = base-stock **+8** each |
| Delays | order 1 + ship 2 |

Capacity 22 is **above** mean exogenous demand (15), so the game is not a pure
“impossible mean” problem. Under scripted adaptive wholesaler + aggressive
retailers it still behaves as chronic **upstream scarcity**:

- Capacity binds ~**42–58%** of weeks on development/validation headline seeds.
- Wholesaler `inventory_on_hand == 0` on ~**58–81%** of weeks (long streaks).
- Early weeks: receive pipeline 4 while incoming retailer claims jump to 20–45 →
  backlog spikes; factory production saturates at 22 while distributor/factory
  order claims climb toward the order cap (128).
- Hub system prompt still advertises capacity 22; that matches the frozen Tier-5
  leaderboard, not a bug in the prompt text.

The aborted 12-update retrain was started **before** research capacity was
raised, so it almost certainly trained under **capacity 22**. Tiny RL gains and
weird baselines are consistent with a mostly capacity-bound local cost surface.

**Do not confuse** `on_hand == 0` with structural unsatisfiability. After raising
research capacity to 400, adaptive fill rates are ~**1.0** with capacity bound
rate **0%**; remaining `on_hand == 0` weeks are often hand-to-mouth (ship ≈
incoming, backlog ≈ 0), not starved backlog growth.

## Env / protocol fix (fulfillment possible)

Research scenarios only (`research_spec` in
`beer_distribution_rl/research/live_y_domain_randomized_grpo_v1/environment.py`):

- **`capacity: 400`** (already landed). Factory can cover bullwhip / large
  claims so the wholesaler can actually clear retailer orders when it orders
  sanely.
- **Initial conditions:** keep Tier-5 defaults `12 / 4 / 4` for continuity with
  topology and prior CRN seeds. Optional later ablation: `24/8/8` or `36/12/12`
  if we want less early transient backlog; not required for feasibility.
- **Aggressive retailers:** keep on (strategic claim inflation + proportional
  rationing). Scarcity is now about **allocation / information / bullwhip**,
  not a hard production ceiling at 22.
- **Frozen Hub Tier-5 leaderboard** (`scenario_for(5, …)` capacity 22) stays
  unchanged. Public static play uses capacity 400 separately.

### Information contract (explicit)

| Fact | Disclosed to model? |
|---|---|
| Demand law / λ / eval process ids | **No** |
| Factory capacity | **No** (still stripped from observation JSON; system prompt does not mention it) |
| Local wholesaler state, retailer orders visible to W, costs, delays, order cap | **Yes** |
| Rival policy parameters / private state | **No** (prompt only notes a scripted rival may order aggressively) |

Hiding capacity remains correct under capacity 400: the number is non-binding
for competent policies, so leaking it would mostly teach a useless constant.

## Training plan (stronger than aborted 12-update attempt)

Reuse demand CRN / 16 train seeds / 16 eval seeds from
`experiments/live_y_domain_randomized_grpo_v1/seed_manifest.json`.

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3.5-4B`, bf16, LoRA r/α 16 on q/k/v/o/gate/up/down |
| Prompt | `research_system_prompt` + `research_observation_user_message` |
| Updates | **12** (cycle 16 train seeds) |
| Seeds / update | **8** |
| Group size | **6** → **576** trajectories, **20,736** decision turns max |
| LR | **1e-5** |
| Generation cap | 32 train / parser ceiling 192 |
| Reward | negative local wholesaler return-to-go; group-mean baseline; protocol fail −1e5 |

Defaults live in `scripts/train_live_y_domain_randomized_grpo_v1.py` (WIP aligned
to this design). **Full GPU launch requires explicit user go-ahead.**

### Pre-launch checklist (cheap)

1. Confirm `research_spec(...).capacity == 400` and adaptive capacity-bound rate
   ≈ 0 on a train seed (unit test / CPU).
2. ✅ Hindsight under capacity 400 published
   (`artifacts/live_y_capacity_400/evaluations/hindsight_perfect_costs.json`;
   mean perfect ≈ **287** vs ~721 under capacity 22). OpenRouter board for
   Laguna / DeepSeek / Nemotron is in the same directory.
3. Optional: one-seed A40 smoke (`--smoke`) only if user wants a cheap GPU check.
4. Then full 12-update run on a fresh pod with network volume.

## Success metrics

Primary (same formula as v1 board):

`100 * hindsight_perfect_mean_local_cost / policy_mean_local_cost`

on the fixed 16-seed research eval set, under the research prompt.

| Gate | Target |
|---|---|
| Protocol-clean | ≥ 95% decisions; 16/16 scored episodes |
| Vs untrained base (research prompt) | **≥ +10 pp** perfect-cost percentage |
| Soft ceiling | within ~1.15× of adaptive / approach hindsight (100%) |
| Feasibility diagnostic (reporting) | mean capacity-bound rate near 0; fill rate not chronically ≪ 1 under adaptive reference |

Prior capacity-22 / capacity-leaking board numbers are **not** comparable and
must not be mixed into the new leaderboard without a footnote.

## Files

| Path | Role |
|---|---|
| `beer_distribution_rl/research/.../environment.py` | research `capacity: 400` |
| `beer_distribution_rl/research/.../prompting.py` | hide demand law + capacity |
| `scripts/train_live_y_domain_randomized_grpo_v1.py` | 12×8×6 schedule defaults |
| `experiments/live_y_feasible_supply_grpo_v2/DESIGN.md` | this note |
| `experiments/live_y_feasible_supply_grpo_v2/evaluation_protocol.json` | eval contract for the feasible-supply run |
| `artifacts/live_y_capacity_400/evaluations/hindsight_perfect_costs.json` | perfect reference under capacity 400 |
