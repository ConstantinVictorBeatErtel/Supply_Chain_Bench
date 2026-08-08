# GPU / compute ledger

Hard budget cap: **$250**. Append a row before every paid GPU job.

| Date | Job | Hardware | Est. hours | Rate ($/hr) | Est. cost ($) | Actual ($) | Notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | Tier 1 is laptop-only; no GPU spend yet |
| 2026-07-15 | exp/bprime-control B×B′ matrix | — (not started) | 0 | 0 | 0 | **0** | **Gate block:** Prompt 2 ablation ⇒ channel not load-bearing; B′ train skipped. Planned slice was {B,B′}×{serial,y}×{∞,1.0μ,0.8μ}×prop×AR1×10seeds @400k. See `artifacts/diagnostics/v11_bprime.md`. |
| 2026-07-15 | **projection only** — LLM GRPO rolling-W8 (order-only, 9 cells × 50 upd × G=4) | 4090 @ $0.50/hr | ~177 (corrected) | 0.50 | **~$89** | **0** | **Not a spend.** Branch `feat/llm-rolling-context` SHA `4debc15…`. tokens/week≈538 (measured W=8 steady) × T=52 × 5 roles × G=4 × 50 upd × 9 cells. Resampling factor **1/(1−p)=1.00** from `llm_text_io.md` final p=0. Naive audit ~$100 @ 600 tok × 1.00 ≈ $99; corrected measured **~$89**. Fits $250 w/ ~$161 margin. See `artifacts/diagnostics/llm_rolling_context.md`. |
| 2026-07-15 | **projection only** — LLM GRPO (order-only, Y×{∞,1.0μ,0.8μ}×prop×3seeds=9 cells) | 4090 @ $0.50/hr | 1740 (full hist, 200 upd) / 200 (roll-W8, 50 upd) | 0.50 | **870 / 100** | **0** | **Not a spend.** Audit `preflight/llm-tier-readiness` SHA `061aa592…`. tokens/week≈1330 (full) or ≈600 (W=8) × T=52 × 5 roles × G=4 × updates × cells. Full-history @200 upd **exceeds $250**; rolling-W8 @50 upd **fits ~$100**. Context retention drives ~54% of tokens vs W=8. See `artifacts/diagnostics/llm_tier_readiness.md` Check 7. |
| 2026-08-08 | Track A artifact recovery from stopped Runpod volume | RTX 4090 | <=0.25 | 0.69 | **<=0.17** | TBD | Resume existing recovery pod only long enough to inventory and retrieve the completed Qwen3.5-4B LoRA run; stop immediately after transfer. No training. |
| 2026-08-08 | Track A artifact recovery start attempt — blocked before provisioning | RTX 4090 | 0 | 0.69 | **0** | **0** | Runpod rejected the start because the account balance was too low; pod remained stopped and no GPU time accrued. The preceding recovery estimate remains applicable after funding. |
| 2026-08-08 | Track A artifact recovery second start attempt — blocked before provisioning | RTX 4090 | 0 | 0.69 | **0** | **0** | Runpod again rejected `s7bbri3e3zilb5` because the account balance was too low; no GPU time or spend accrued. Recovery artifacts remain inaccessible until the account is funded. |
