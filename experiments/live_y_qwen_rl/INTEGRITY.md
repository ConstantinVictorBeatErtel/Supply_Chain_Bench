# Live-Y post-training evaluation integrity

*Frozen: 2026-08-08*

This note governs only the post-training comparison in
`experiments/live_y_qwen_rl/`. It does not amend the environment v0.2.0
specifications, grader, scenario parameters, existing published results, or the
separate 20-week serial benchmark.

## Seed audit and frozen roles

The prior manifest contained 16 `train` seeds and 10 `eval` seeds. Recomputing
each value from `SHA-256('beer-live-y-rl-v1|<label>|<index:05d>')[:16]`
confirms every value, and set intersection confirms the two sets are disjoint.
Schema 2.0 retains those exact values under the clearer roles `train` and
`research_development`.

The new `held_out_test` split contains 20 values derived with the previously
unused label `beer-live-y-rl-v1|held_out_test|<index:05d>`. Before freezing it:

- all reachable local and `origin/*` branch histories were inspected; the live
  split had only the two identical introduction commits (`3a3b590` and
  `d7c07c8`), and `git log --all -S` found no prior held-out-test namespace or
  first candidate value;
- the candidate hex values were compared with all 18 normative environment
  development/validation/test seeds and the 100-seed Tier-5 Y frontier split;
- the same 64-bit values were compared numerically with the 100-seed serial
  held-out split and its 100-seed robustness replication; and
- all three live-Y sets were checked for internal uniqueness and pairwise
  disjointness.

The regression suite repeats the manifest comparisons. The result is zero
overlap in every comparison. “Held out” here means never used for model
selection, not secret: the manifest is committed for reproducibility.

The allowed-use matrix is enforced before task construction or CUDA loading:

| Purpose | Only permitted split |
|---|---|
| RL rollout or SFT/rejection-sampling data generation | `train` |
| calibration, development evaluation, prompt tuning, hyperparameter selection, early stopping | `research_development` |
| one-shot final report after model and protocol freeze | `held_out_test` with explicit opt-in |

An SFT dataset that contains any recognized live-Y seed is rejected unless all
rows are training-seed rows. `--require-live-y-train-seeds` additionally
requires provenance on every SFT row. Held-out-test access in the live-Y runner
requires both `--mode held-out-test` and `--allow-held-out-test`; the flag is
invalid in other modes.

## Forensic correction to the prior development summary

The historical 16/10 seed sets themselves are sound, but source revision
`221462a` did not use them as reported for every row. In the inherited pilot,
`main()` constructed `train_tasks`, ran both `eval_pre_development` and
`eval_post_development` on that same object, and constructed the ten evaluation
tasks only in `--eval-only` mode. The live-Y wrapper returned the 16 train seeds
while in training mode. Therefore the completed training process's Qwen
pre/post diagnostics were training-seed evaluations.

The published compact values contain independent numeric evidence of mixed
provenance:

- recomputation on the 10 `research_development` seeds gives naive cost
  `1284.8 ± 72.6738223` and adaptive-base-stock cost
  `601.6 ± 72.5358072`, exactly matching the two reference rows;
- recomputation on the 16 `train` seeds instead gives naive `1604.5625` and
  adaptive base-stock `917.09375`;
- the Qwen pre/post means `1579.6875` and `1096.3125` have exact sixteenth
  granularity and are the values saved by the source path that evaluated all 16
  `train_tasks`.

Consequently, the reported 30.60% pre/post reduction is a **training-seed
diagnostic**, not a development result. The reported 0–100 Qwen scores combine
the 10-seed naive denominator with 16-seed Qwen costs and are invalid
cross-split comparisons. The existing `RESULTS.md` and `summary.json` are
published reference artifacts and are intentionally unchanged. Recovered raw
evaluation files may add evidence, but must not be presumed to repair this
provenance issue without verifying their per-episode seed IDs.

Future training now constructs 16 rollout tasks and a separate 10-task
research-development set; both pre- and post-training evaluations use only the
latter. Focused tests exercise the routing directly.

## One policy interface

`evaluation_protocol.json` freezes one comparison cell and action protocol:

- wholesaler only in environment v0.2.0 `t5-strategic-y-v2`, headline Y
  topology, 36 operational decisions;
- the unmodified `BeerEpisode` constructs identical deterministic scripted
  counterparties for every policy;
- orders are integers from 0 through 128;
- every model receives the same system prompt and role-local observation using
  the repository's JSON action format;
- the response must parse as one JSON object with exactly the key `quantity`
  and an integer value in range; bare integers, tool-call text, floats, extra
  keys, repairs, and fallback orders are invalid protocol attempts; and
- after 36 valid actions, `BeerEpisode` performs its native three settlement
  weeks and terminal exposure accounting.

This is a Tier-5 Y local-wholesaler-cost leaderboard. It must not be pooled
with or used to overwrite the frozen 20-week serial benchmark in
`eval/run_eval.py` and `eval/held_out_seeds.json`.
