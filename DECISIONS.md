# Design decisions

Logged defaults for the v1 research codebase. Change only with a dated note.

| Decision | Default | Rationale |
|---|---|---|
| Python | 3.11+, package `beer_distribution_rl` | Spec / modern typing |
| Layout | Code under `beer_distribution_rl/{env,agents}/` | Enables `import beer_distribution_rl.env` |
| Classic eval horizon | T=36 | DQN-paper compare (Oroojlooy et al.) |
| Train default horizon | T=52 | PROJECT_SPEC §3.1 |
| Lead times | ship delay L_s=2, order delay L_o=1 | PROJECT_SPEC §3.1 |
| Action | absolute non-neg int order ∈ {0…128} | v1.1 (2026-07-12): raised from 64 per B1; AR(1)+relative Δ ratchets rarely bind |
| Classic costs | h=0.5, b=1.0 uniform all roles | Classic beer-game / DQN config |
| Classic demand | step 4→8 at week 5 (weeks 1–4 demand=4, then 8) | Classic MIT step |
| Training demand (v1.1) | AR(1) φ=0.7, μ=7.5, σ=2.0 (default); also `uniform`, `regime_switch`, `classic_step` | D6: U[0,15] lag-1 R²≈0; AR(1) R²≈0.47 — channel needs something to carry |
| Initial state | on-hand=12, backlog=0, pipelines filled with 4s | DQN-repo convention |
| Base-stock levels (DQN gate) | S=[9,5,3,1] retailer→factory | Oroojlooy et al. arXiv:1708.05924 §4 |
| DQN paper gate config | demand U{0,1,2}, L_s=L_o=2, ch=[2,2,2,2], cp=[2,0,0,0], S=[9,5,3,1], init 0 | Oroojlooy et al. §4 parameters |
| Base-stock cost target (calibrated) | mean cost/period ∈ **[4.5, 7.5]** under our Sterman-compatible event order | Empirically ~5.9 with S=[9,5,3,1]; published 2.008 uses their order-before-receive semantics — we keep Sterman week order (DECISIONS) and treat 2.008 as external reference, not a hard match |
| Gate also requires | base-stock ≪ Sterman on same DQN config (≥5×); retailer cost share ≥80% | Shape matches published allocation [1.91,0.05,0.02,0.03] |
| Research classic config | step 4→8, h=0.5, b=1.0, L_s=2, L_o=1 | PROJECT_SPEC §3.1 |
| Inventory position | on-hand − backlog + on_order (on_order += placed, −= received) | Required for correct base-stock |
| Sterman params | α_s=α_sl=0.5, β=1.0, θ=0.36 (demand smoothing) | Sterman 1989 anchoring-and-adjustment; documented for reproducibility |
| Week event order | receive shipments → receive orders → fill/ship → place orders → accrue costs | Sterman-compatible |
| Factory under C=∞ | production order enters factory ship pipeline (lead L_s) | Unlimited external supply |
| M1 scope | classic core + baselines + signals/rationing + PettingZoo; no IPPO/GRPO | Validation gate before training |
| Test stack | pytest + hypothesis; env has zero ML deps | Hub-ready pure Python env |
| Perf target | >10k `BeerGameCore.step()`/sec on laptop | PROJECT_SPEC §6 |

## Emergence constraints (non-negotiable)

- Regimes A/B: one policy (or LoRA) per role; no shared parameters or critics.
- Regimes A/B rewards: strictly local costs; no system-cost term, no cooperation shaping, no honesty reward.
- Signaling: optional and unverified; honesty is measured, never rewarded.
- Regime C (shared system reward) exists only as reproduction of arXiv:2605.17036.

## M2 — IPPO

| Decision | Default | Rationale |
|---|---|---|
| Algorithm | Independent PPO (CleanRL-style), MLP 2×256 | Spec Tier 1; auditable; laptop-friendly |
| Parameter sharing | Forbidden in A/B; also forbidden in C for M2 | Emergence claim; C only shares the *reward*, not weights |
| Shared critic | Forbidden — one critic head per role | Spec non-negotiable |
| M2 regimes | A and C only on classic step demand | Gate: cost ballpark vs Sterman/base-stock |
| Regime B / capacity matrix | M3: Regime B × caps × rationing; relative claims; honesty measured only | Phase diagram before GPU spend |
| M3 train timesteps (sweep default) | 50k / cell, ≥10 seeds on headline B cells | Laptop-feasible; extend later |
| Serial rationing equivalence | On serial topology, prop/uniform/honesty-weighted coincide (one claimant) | P3 needs Y-topology; documented in M3_REPORT |
| M3 matrix completed | B × 5 caps × 2 ration × 10 seeds × 50k steps | Phase diagram v1; P1 weak, P2 mixed, P3 N/A |
| Action space | relative: order = clip(demand + Δ, 0, 128), Δ ∈ [-8, 8] | v1.1: cap 128 (B1); relative Δ retained for tractability |
| Reward scale | 0.1 (training only; logged costs unscaled) | Stabilizes value learning |
| Reproducibility | YAML + seed + git SHA in `run_meta.json` | Spec §6 |
| Demand info value | logged to `demand_info_value.json` / `run_meta.json` at train start | Paper justification for env v1.1 demand change |
| Boundary actions | `eval/frac_actions_at_cap`; warn if >5% | Healthy runs ≈0; saturation contaminates sensitivity |

## Tier-1 v1.1 matrix (Agent R1)

| Decision | Default | Rationale |
|---|---|---|
| Matrix axes | {A,B,C} × {serial,y} × {∞,1.2μ,1.0μ,0.8μ} × {prop,uniform,honesty_w} × {ar1,regime_switch} × ≥10 seeds | Close D1 (A vs B); P3 needs Y; informative demand |
| Dropped cap | 1.5μ removed from M3 grid | Concentrate on binding region; ∞ retained as slack anchor |
| Prune: C=∞ rationing | Keep **proportional only** at ∞ | Capacity never binds ⇒ rationing policies identical |
| Prune: serial rationing | Keep **proportional only** on serial | Single claimant ⇒ prop≡uniform≡honesty_weighted (DECISIONS M3) |
| Kept after prune | **840** cells (was 1440 full cartesian @ 10 seeds) | `prune_summary.json` written each run |
| Y × ar1 demand | `CorrelatedYDemand` (shared factor) | Rival broadcasts informative; matrix label stays `ar1` |
| Y × regime_switch | `TwinCustomerDemand(RegimeSwitch)` | Independent streams; ablation vs correlated factor |
| Cell parallelism | 8–16 processes (`ProcessPoolExecutor`) | Dominant speedup; CPU-bound MLPs |
| Vec envs / cell | `n_envs=64`, `rollout_steps=128` (batch 8192) | GPU idle otherwise; no JAX rewrite |
| Timesteps / cell | **400k** (was 50k in M3) | With vec envs, 50k ⇒ only ~6 PPO updates; 400k/(64×128)≈49 matches M3 learning budget |
| Device | cpu default; `cuda`/`mps` via CLI/Colab | Move nets to GPU only after vec envs |
| Resumability | skip cell if `final_eval.json` exists | Colab sessions die; checkpoint per cell |
| Bind event logs | `eval/frac_capacity_binds`, `eval/frac_allocation_triggers`, `week_events.json` | Closes D5 log gap (no checkpoint recompute) |
| Artifacts | `config.yaml` + `run_meta.json` (seed, git SHA) per run | Spec §6 |
| Runner | `scripts/run_tier1_matrix.py`; Colab: `notebooks/colab_tier1.ipynb` | Coffee-break target ≪15 min on Colab |

## B′ control / M4 gate (2026-07-15) — `exp/bprime-control`

| Decision | Default | Rationale |
|---|---|---|
| B′ train matrix | **Skipped** (0 h / $0) | Prompt gate: only train B′ if Prompt 2 ablation shows channel plausibly load-bearing |
| Prompt 2 result | Channel **not** load-bearing; shuffles inert under det. eval | B scarcity gap largely eval-mode confound; see `v11_ablation.md` / `v11_bprime.md` |
| Causal prior without B′ | Treat **B ≈ B′** as the supported framing | Babbling / negative-result writeup; do not claim information-flow advantage |
| **M4 LLM gate** | **Not cleared** | No demonstrated load-bearing cheap-talk channel; revisit B′/M4 only after a checkpoint fails shuffle-time |

## Recurrent IPPO baseline (2026-07-15) — `baseline/recurrent-ippo`

| Decision | Default | Rationale |
|---|---|---|
| LLM-vs-MLP comparison | **Memory-matched** | Recurrent GRU IPPO retains own T=52 history matching planned LLM context (Check 4); removes memory confound from language-prior claims |
| Recurrent architecture | GRU(obs→128) + 2×256 actor/critic per role | Preferred over frame-stack; fits R1 runner; E1 own-history only |
| Headline cells | Regime A × {serial,Y} × {∞,1.0μ,0.8μ} × prop × AR(1) × 10 seeds (+ Y×uniform for gaming) | Order-only setting matching LLM channel drop |
| Memory-only finding | GRU does **not** cut cost vs Markovian under matched budget; shortage gaming **survives** | See `artifacts/diagnostics/recurrent_baseline.md` |

## LLM text I/O (2026-07-15) — `feat/llm-text-io`

| Decision | Default | Rationale |
|---|---|---|
| Product path | `beer_distribution_rl/agents/llm/` | Clears readiness blocker 2; provisional smoke harness is not a substitute |
| Information set | Own history only (Check 3 / recurrent match); order-only (no cheap talk) | E1 no-leak in text; apples-to-apples vs GRU baseline |
| Action emit | Relative Δ ∈ [-8, 8] → clip(demand+Δ, 0, 128) | IPPO parity (DECISIONS M2 / Check 0) |
| Decoding | **JSON-schema constrained** (Ollama `format` / GBNF twin for vLLM); resample≤3 logged | Check 5 regex parse-fail 30–39% → **0%** uniform across cap×role; see `llm_text_io.md` |
| Resampling multiplier | **1.00** (= 1/(1−p) at p=0) | Feeds corrected GRPO budget; drop ~1.5× parse tax from Check 7 drafts |
| Spend on this branch | **$0** / no GRPO | Inference-only local Qwen2.5-3B smoke |

## LLM rolling context (2026-07-15) — `feat/llm-rolling-context`

| Decision | Default | Rationale |
|---|---|---|
| Context retention | **Rolling own-history window W=8** (`DEFAULT_ROLLING_WINDOW`) | Clears readiness blockers 3+7; full history fits 32k but GRPO full-hist ~$870 over $250; W=8 ≈$89 corrected |
| Ablation knob | `AgentMemory.window` / `serialize_prompt(window=)` | One-line W-sensitivity later (mirrors recurrent “does memory help?”) |
| Per-prompt tokens @ W=8 | **~538 mean / 540 max** (chars/4) | ≪ Qwen2.5-3B 32k; persistence PASS on T=52 |
| GRPO budget (projection) | **~$89** after ×**1/(1−p)=1.00** (p=0 from `llm_text_io.md`) | 9 cells × 50 upd × G=4 × measured W8 tokens × 4090 @$0.50/hr; fits $250 w/ margin; **$0 spent / no GRPO** |
| E1 | Own history only in prompt | No other-role private state; leak hits 0 |

## LLM-tier readiness v2 (2026-07-15) — `preflight/llm-tier-readiness-v2`

| Decision | Default | Rationale |
|---|---|---|
| Baseline SHA | `76ce8a1f978f39c88667b518240d5187879de13e` | `git rev-parse HEAD` at branch tip (= `main` after recurrent-baseline + text-I/O + rolling-context merges) |
| Blocker 2 (text I/O) | **GO** | Product serializer+parser; parse-fail **0%** uniform cap×role (`llm_text_io.md`) |
| Blocker 3 (context) | **GO** | Rolling W=8 productized; T=52 persistence PASS; ~538 tok ≪ 32k (`llm_rolling_context.md`) |
| Blocker 4 (memory) | **GO** | Recurrent IPPO baseline exists; shared own-history info set; LLM W=8 not strictly more memory than GRU (`recurrent_baseline.md`) |
| Blocker 7 (budget) | **GO** | Corrected W=8 × 1.00 resampling ≈ **$89** / 9 cells fits $250 (`LEDGER.md`) |
| Non-negotiables | **GO** | One LoRA/role; local-cost GRPO scalar; no B signaling; no C system reward |
| **Overall readiness verdict** | **READY-FOR-FIRST-GRPO-CELL** | All v1 blockers cleared; details: `artifacts/diagnostics/llm_tier_readiness_v2.md` |
| First cell (SPEC only) | Y × Regime A order-only × **1.0μ** × prop × AR(1) × **3 seeds** × Qwen2.5-3B × LoRA/role × W=8 × G=4 × 50 upd | Smallest probe of RL vs order-signal/shortage-gaming; est. **~$30** (3/9 of $89); kill if no RL signal / collapse / protocol breach |
| GRPO | **Not started** | $0 this audit; launch only after human review |

## Shortage gaming / order-stream pivot (2026-07-15) — `analysis/shortage-gaming`

| Decision | Default | Rationale |
|---|---|---|
| Cheap-talk as strategic object | **Demoted** | `v11_signal_content`: babbling; ablation: channel not load-bearing |
| Primary strategic object | **Order stream** under multi-claimant rationing (Y) | Orders costly; proportional rationing rewards claim inflation (Lee et al. 1997) |
| Eval protocol | Matched-deterministic (`greedy=True`, seed+10k) | Same definition as eval-mode blast radius; no train/reward/env edits |
| Inflation benchmark | Base-stock S=30 (sens. {9,22,30,45} + pass-through) | Installation stock for AR(1) μ=7.5, L=3 (BUGHUNT); not classic DQN S=9 |
| Gaming label rule | Require response to **both** capacity tightness **and** rationing rule | Guard against mechanical / cap-128 artifacts |
| Headline verdict | **`supported`** | B×Y×AR(1): gap(∞→0.8μ)=+8.80 under prop; mean prop−uniform @ tight = +6.79; serial scarcity Δ=−4.82 (no-rival control). Details: `artifacts/diagnostics/shortage_gaming.md` |
| Rival externality | Weak / mixed | Share-based corr(Δorder,Δalloc) prop 0.24 vs uni 0.13 at 0.8μ — directional but not decisive |
| Channel dependence | A ≈ B on Y×prop inflation | Gaming does not require the broadcast channel |

## Honesty-weighted recheck (2026-07-15) — `analysis/honesty-weighted-recheck`

| Decision | Default | Rationale |
|---|---|---|
| Logged HW mechanism | **Broadcast** truthfulness EMA (not orders) | `measure_honesty` on claimed_demand/inventory; orders never enter EMA |
| Share-drop under matched-det | **Survives** | All-role HW ~0.38 vs prop ~0.51; **retailer** share → ~0 (argmax silence) vs prop ~0.28 — claimants flee; upstream still broadcasts |
| EMA-never-accumulated artifact | **Ruled out** | Det EMA flat only because silent; stoch probe: EMA≠0 frac≈0.62, mean\|ΔEMA\|≈0.76 |
| Headline grade | **`footnote-grade`** | Real disengagement from a noise-weighted reputation game; not P3 truth restoration. Order-truthfulness re-run is the interesting pivot. Details: `artifacts/diagnostics/honesty_weighted_recheck.md` |

## Eval-mode blast radius (2026-07-15) — `diag/eval-mode-blast-radius`

| Decision | Default | Rationale |
|---|---|---|
| Logged Tier-1 A−B scarcity gaps (28–53% and siblings from `final_eval` / `index.json`) | **RETRACTED** | Root cause: `IPPOTrainer.evaluate` uses `greedy=not self.signaling` — A/C argmax, B stochastic. Not a fair cross-regime compare. Corrected matched-deterministic table: `artifacts/diagnostics/eval_mode_blast_radius.md` (baseline SHA `061aa59235397b7360c32a01cf4f98add0dd503a`) |

## LLM-tier readiness (2026-07-15) — `preflight/llm-tier-readiness`

| Decision | Default | Rationale |
|---|---|---|
| Baseline SHA | `061aa59235397b7360c32a01cf4f98add0dd503a` | `git rev-parse HEAD` at audit branch tip (= `main`) |
| LLM action schema | **ORDER only** (cheap-talk / broadcast **dropped**) | Channel babbling + not load-bearing; shortage-gaming lives in the order stream (`analysis/shortage-gaming`) |
| Context | Retain **structured own-history** across weeks | Design under audit; full history fits 32k; rolling window needed for **budget** |
| **Overall readiness verdict** | **BLOCKED-ON-{product-text-I/O, parse-fail productization, memory-confound protocol, GRPO budget plan}** | Check 1 phenomenon **GO** (`supported`); Checks 2–3 product gaps; Check 5 floor GO-with-caution (3–4× base-stock, parse-fail ~30–40%); Check 7 full-hist GRPO over $250. Details: `artifacts/diagnostics/llm_tier_readiness.md` |
| GRPO | **Not started** | $0 spend; no training code in this preflight |

## M4 capability-floor smoke — Qwen2.5-3B (2026-07-17) — `scripts/run_llm_episode.py`

| Decision | Default | Rationale |
|---|---|---|
| Capability floor (Qwen2.5-3B, greedy, JSON-schema decode) | **NOT CLEARED** | Colab smoke, T=52: parse-fail 0% both cells, but mean order collapses toward 0 (classic 0.38/0.08/0.0/0.0; y_tight 1.88/1.77/0.87/0.31/0.02) instead of tracking demand. Classic system cost 3.5× the naive demand-matching baseline (9636 vs 2768); y_tight ≈ parity but via backlog spiral, not real ordering. |
| Action on floor failure | **Do not launch GRPO on 3B** | PROJECT_SPEC §4 Tier-2: "if the base model can't play coherently zero-shot, results at that size are uninterpretable — move up a size." Re-run the same free smoke on 7B (or an alt. base model) before any paid step. |
| Spend | **$0** | Colab free-tier T4; no GPU rental used |

## Hub environment interface v1 (2026-07-18) — `codex/verifiers-environment`

| Decision | v1 choice | Rationale |
|---|---|---|
| Agent exposure | **One controlled role per rollout** against deterministic scripted counterparties | Supports attributable, inexpensive, exactly replayable model comparisons; true multi-agent is deferred |
| Role coverage | Separate tasks for every serial and Y role | Prevents retailer-only results from hiding echelon-specific difficulty |
| Action | Strict `place_order(quantity: integer)` tool, absolute quantity in [0, 128] | Native agent interface; invalid actions are rejected without clamping, defaults, state advance, or RNG consumption |
| Decision boundary | Observe receipts, current demand/order, fulfillment, and local state **before** ordering | Matches the Beer Game decision rather than the legacy caller's action-before-new-observation behavior |
| Observation | Current local state plus last 8 own-role records; no other-role state or future delayed orders | Fixed memory contract with bounded context and no `order_pipeline` look-ahead leak |
| Horizon | 36 weeks default; 52-week stress setting | Preserves delayed dynamics while keeping frontier-model evaluation affordable |
| Objective disclosed to agent | Minimize controlled-role local cost | Preserves the original self-interested-agent question; exact grading is Stage 3 |
| Distribution | Verifiers reference adapter; optional HUD v6 parity adapter later | Prime Intellect is the publication target; HUD demonstrates portability without entering the simulator core |
| Normative specification | [`docs/ENVIRONMENT_SPEC.md`](docs/ENVIRONMENT_SPEC.md) | Stage 2 approved 2026-07-18 |

## Hub environment grading v1 (2026-07-18) — `codex/verifiers-environment`

| Decision | v1 choice | Rationale |
|---|---|---|
| Headline outcome | Controlled-role undiscounted local total cost | Preserves the self-interested-agent question and remains directly auditable |
| Hub scalar | `C_base / (C_base + C_agent)` using frozen same-seed base-stock reference | Bounded, monotonic, cross-cell normalization; base-stock anchors at 0.5 without clipping |
| Horizon defense | Deterministic settlement plus one-period terminal inventory-position charge | Advances committed pipelines and values remaining shortages/commitments without extra model calls or randomness |
| Protocol | Any invalid first attempt makes official episode reward zero | Prevents deliberate retries from purchasing extra inference; repaired costs remain diagnostic only |
| Diagnostics | Local/system cost, externality, immediate and horizon service, bullwhip, normalized order volatility | Exposes shortage gaming and constant-zero policies without arbitrary weighted reward terms |
| Constant demand | Bullwhip is `null`; report normalized order volatility | Variance-ratio bullwhip is undefined when demand variance is zero |
| Reasoning quality | No reasoning term in v1 reward; optional offline trace-grounded strategy audit only | Post-hoc prose is not hidden reasoning and judge variance should not contaminate programmatic scoring |
| Normative specification | [`docs/REWARD_SPEC.md`](docs/REWARD_SPEC.md) | Stage 3 approved 2026-07-18 |

## Hub environment difficulty ladder v1 (2026-07-18) — `codex/verifiers-environment`

| Decision | v1 choice | Rationale |
|---|---|---|
| Tier 1 | Constant demand 8, serial, shipment notices | Isolates protocol and steady delayed control |
| Tier 2 | Stationary AR(1): μ=7.5, φ=0.7, σ=2.0 | Adds persistent uncertainty without changing topology or observability |
| Tier 3 | Same AR(1) with hidden seeded shift: week {15,19,23}, μ after {4,12} | Tests change detection in both directions and prevents calendar-only policies |
| Tier 4 | Exact Tier 3 latent trajectory with shipment slots hidden | Paired contrast isolates belief-state maintenance under partial observability |
| Tier 5 | Y, correlated demand, calibrated capacity 22, proportional rationing, aggressive rival +8 | Tests strategic robustness under intermittent scarcity and a competing claimant |
| Tier 5 controls | Base-stock rival/proportional and aggressive rival/uniform | Makes claims about opponent pressure and allocation incentives falsifiable |
| Counterparties | Deterministic `adaptive_base_stock_v2` with EMA forecast α=0.25 and target `L*forecast` | Uses the three decision intervals proven by the order-to-receipt timing test, without privileged future knowledge |
| Frontier compute | All models: five-tier retailer screen, 10 seeds; at least two models: all roles, 5 seeds | Preserves broad comparison while bounding model calls; all unpromoted results remain visible |
| Seed splits | 3 development, 5 validation, 10 test; 64-bit SHA-256 seeds serialized as hex strings | Reproducible splits without sequential-seed, mutable-RNG coupling, or JSON precision loss |
| Normative specification | [`docs/DIFFICULTY_LADDER.md`](docs/DIFFICULTY_LADDER.md) | Stage 4 approved 2026-07-18 |

## Hub implementation API (2026-07-18) — `codex/verifiers-environment`

| Decision | Choice | Rationale |
|---|---|---|
| Verifiers version | Pin stable `verifiers==0.2.0` | PyPI and official tag inspection showed 0.2.0 supersedes the 0.1.x APIs |
| Native package contract | Export exactly one typed `BeerTaskset` and bundled `BeerHarness` through `__all__` | Verifiers 0.2 v1 discovers plugin classes; legacy `load_environment()` is explicitly prohibited for new packages |
| Harness | Custom MCP-only rolling-context harness | Built-in minimal harness retains the growing conversation; that would violate the approved 8-record model-memory contract |
| Package boundary | Self-contained `environments/beer_distribution_game/` Hatch project | Matches Hub installation/push unit and keeps Verifiers out of the framework-neutral simulator modules |
| Tier 5 capacity calibration | **22**, replacing the provisional 15 | Rechecked after the v2 timing correction on five validation seeds: 22 binds in 100/180 base-rival weeks (55.6%, inside the predeclared 10%--70% gate), aggressive retailers have 0 order-cap hits, and base-stock beats random. No test split or new frontier result was inspected. |

## Superseded v1 Hub smoke (2026-07-18) — `codex/verifiers-environment`

| Decision | Choice / finding | Rationale |
|---|---|---|
| Akash model | `deepseek-ai/DeepSeek-V4-Flash`, temperature 0 | Cheapest account-visible tool-capable model; direct strict-tool probe passed |
| Tool enforcement | API-level `tool_choice="required"`, parallel calls disabled | Initial integration run exhausted 64 tokens on prose and terminated at the protocol gate; this was a harness failure, not a model result |
| Provider resilience | Two request-level retries, 120-second request timeout | Akash returned intermittent HTTP 502 responses; retry occurs before state mutation and completed the same episode cleanly |
| Concurrency | 1 for Akash smoke | Five concurrent episodes stalled for 12 minutes with no completion; sequential execution completed reliably |
| Five-tier result | Protocol-clean on all five development-seed-0 tasks; rewards T1–T5: 0.816, 0.383, 0.225, 0.224, 0.171 | Preliminary capability smoke only; one seed, no variance, no test split |
| Successful-trace usage | 184,727 input + 7,920 output tokens; estimated $0.028 at recorded price | Excludes the direct probe and aborted integration attempts; not an account-billing total |
| Base-stock timing correction | **Resolved in environment v0.2.0** | A one-unit impulse placed after week 1 demand arrives at the retailer at the start of week 4. The v1 target double-counted a review period; v2 uses `L*forecast`, orders 8 throughout T1, and has exact graded cost 69 including startup, settlement, and terminal exposure. The v1 summary remains here for auditability; obsolete raw traces were removed from the current tree and remain recoverable from Git history. |

## Corrected v0.2.0 evaluation gate (2026-07-18) — `codex/verifiers-environment`

| Decision / finding | Result | Interpretation |
|---|---|---|
| Corrected retailer smoke | DeepSeek V4 Flash cost 69, paired base-stock 69, reward 0.500; 36/36 protocol-clean actions | Confirms the v2 reference and model/tool path agree on the steady-flow solution |
| Y wholesaler development evaluation | Three seeds; model cost 1,111.8 ± 213.2, paired base-stock 850.7 ± 326.1, reward 0.423 ± 0.060; 108/108 protocol-clean actions | Model lost to base-stock on every seed, supporting the wholesaler as the more informative learning target without claiming held-out generalization |
| Reproducibility boundary | Environment/action replay is exact; separate temperature-0 seed-0 model runs scored 0.451 and 0.449 | Provider generation is not byte-deterministic, so report action traces and distinguish seed variance from repeat-generation variance |
| Recorded corrected-pass usage | 187,974 input + 7,920 output tokens across five successful rollouts; about $0.029 at observed prices | Excludes provider-side billing for failed/retried calls and is not an account statement |

## Teacher-free Qwen3.5-4B live-game RL split (2026-08-02)

| Decision | Choice | Rationale |
|---|---|---|
| Environment | Exact v0.2.0 Tier-5 Y `headline`, wholesaler, 36-week scenario | Matches Hub Tier-5 (capacity 22). The public static game later raised playable capacity to 400; see `live_y_feasible_supply_grpo_v2`. |
| Experiment seeds | 16 train + 10 evaluation master seeds, frozen in `experiments/live_y_qwen_rl/splits.json` | Existing development and validation seeds are the eight public browser-game seeds; normative test seeds remain test-only. The experiment split is SHA-256-derived and does not alter the normative split contract. |
| Training signal | Group-relative negative terminal local wholesaler cost | No demonstrations, action labels, base-stock reward, or system-cost reward enter optimization. |
| Reference policy | Paired adaptive base-stock only during evaluation reporting | Retains an interpretable comparison without acting as a teacher. |

## Canonical serial leaderboard refresh (2026-08-04)

| Decision / finding | Result | Rationale |
|---|---|---|
| Public leaderboard scope | One 100-seed, 20-week serial wholesaler condition | Prevents incomparable live-Y, robustness, and serial experiments from being presented as one ranking. |
| Claude policy | `anthropic/claude-opus-5`, zero-shot | Provides a current frontier reference on exactly the Qwen held-out split. |
| Claude result | Mean total cost 1,754.48 ± 61.14; score 49.66; 0.0% format failures | Same frozen seeds and score formula as Qwen; result is recorded in `results/baseline.json`. |
| Cost controls | Compact state tuple, strict JSON, reasoning disabled, 16-token cap, same-week deduplication, response cache, checkpoint resume | A warm rerun completed 1,512 requests with 1,500 cache hits for $0.02, without GPU infrastructure or changing the decision condition. |
