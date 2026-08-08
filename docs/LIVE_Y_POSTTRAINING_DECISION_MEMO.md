# Tier-5 Y post-training decision memo

**Date:** 2026-08-08  
**Scope:** Qwen/Qwen3.5-4B, wholesaler only, native v0.2.0 Tier-5 Y
`headline`, 36 decision weeks plus native settlement. This is a research memo,
not a change to the frozen environment or reward contract.

## Decision

Run the smallest informative comparison as a four-arm design: untrained base,
the recovered teacher-free group-relative RL adapter, self-generated SFT-only,
and the same self-generated SFT warm start followed by group-relative RL. Do not
add SAO to the executable matrix yet. In arXiv:2607.07508, SAO means
**Single-Rollout Asynchronous Optimization**, not Self-Alignment Optimization.
The paper does not provide author code, data, checkpoints, or enough
implementation detail for a faithful one-A40 port. Public implementations found
during this review explicitly describe themselves as independent or unofficial
reproductions.

The recovered adapter remains an important arm, but its reported 30.60% cost
reduction and 53.96 score are **not yet clean evaluation evidence**. The current
training wrapper selects the 16 training tasks for its in-run pre/post
evaluation, whereas the published naive and adaptive-base-stock rows correspond
to the 10 research-evaluation tasks. Combining those rows makes the published
score cross-split. Re-evaluate base, references, and recovered adapter together
through the frozen policy interface before using the prior result to choose a
method or checkpoint.

## Method comparison

| Method | Supervision and objective | What it answers | Decision |
|---|---|---|---|
| Base Qwen, no training | None | Required anchor for cost, protocol, and order volatility | Evaluate once per frozen reporting split |
| GRPO from base | No demonstrations or action teacher; sampled trajectories receive group-relative advantage from negative terminal local wholesaler cost | Can programmatic selfish reward alone improve the policy? | Keep the recovered adapter; cleanly re-evaluate it. A fresh rerun is needed only if exact compute matching cannot be reconstructed from its artifacts |
| SFT-only | Recommended main corpus: Qwen's own protocol-clean, verifier-ranked best-of-N trajectories on **training seeds only** | Does cloning the base model's own successful behavior provide most of the gain without online RL? | Run |
| SFT → GRPO | Branch from the SFT-only arm's half-budget checkpoint, then use the identical local-cost RL objective | Does SFT provide a useful warm start while RL corrects imitation and selection bias? | Run; this is the leading candidate if SFT-only plateaus |
| SAO | Single-rollout asynchronous actor-critic with rollout-policy log probabilities, strict double-sided token masking, a pretrained value model, two critic updates per actor update, frozen-attention critic training, and skip-observation token-level GAE | Does an asynchronous single-rollout critic beat grouped updates? | Research-only; no training in this phase |

The existing local "GRPO" trainer is more precisely a small group-relative
PPO/GRPO-style pilot: it subtracts the per-seed group mean, broadcasts terminal
advantage to the episode's action completions, and applies one clipped PPO pass
over mean completion log-probabilities. It is not a drop-in reproduction of all
details in the original GRPO paper. Preserve that implementation across the RL
arms for this comparison; changing to TRL/verl at the same time would confound
warm start with optimizer implementation. If canonical GRPO is later adopted,
restart every RL arm under it.

## SFT provenance must be explicit

Two defensible datasets answer different questions and must never share the
same label:

- **Scripted teacher supervision:** actions from adaptive base-stock, MPC, or
  another scripted controller are teacher labels. This is useful as a separate
  imitation/ceiling control, but it injects the reference policy and can suppress
  behavior the emergence study is meant to discover. The repository's existing
  `data/wholesaler_base_stock_train.jsonl` is exactly this kind of teacher data,
  and it is also from the incompatible 20-week serial benchmark; it must not be
  reused for the native 36-week Y comparison.
- **Self-generated/rejection-sampling supervision:** sample complete trajectories
  from the unadapted Qwen policy, reject protocol failures, rank only with the
  frozen deterministic local-cost verifier, and retain the best trajectories.
  This has no external action teacher, but it is still offline SFT on selected
  labels, not teacher-free on-policy RL. Record candidate count, selection rule,
  policy hash, temperature, RNG seeds, and rejected trajectories.

Use self-generated/rejection-sampling SFT for the main four-arm matrix because
it tests the value of a supervised warm start without handing the model the
adaptive-base-stock solution. Add a **separately named** scripted-teacher SFT
control only after the main matrix if self-generated SFT has no signal or if an
imitation ceiling is scientifically useful.

## Why SAO is a poor fit for this first matrix

The cited paper targets asynchronous actor-learner systems with variable-length
agentic coding/reasoning rollouts. Its efficiency case is avoiding group
stragglers, and its online-learning case deliberately changes the reward
preference during training. This Beer Game is deterministic conditional on seed
and actions: every protocol-clean trajectory has exactly 36 short decisions,
the counterparties and reward are stationary, and the planned one-A40 runner
colocates rollout and optimization rather than maintaining asynchronous actors.
There is therefore little straggler or policy-lag problem for SAO to solve.

SAO could eventually be interesting for credit assignment: a state-dependent
critic may be better than broadcasting one terminal group advantage across all
36 decisions, and single-rollout sampling avoids zero-variance groups. Against
that possibility, this setting has only 16 repeated training seeds, a sparse
terminal return, eight-token actions, and a 4B backbone. A value model can
overfit and adds substantial memory/compute. The paper itself cautions that its
30B-A3B results may not transfer to smaller models, dense rewards, or shorter
rollouts. Its frozen-attention critic recipe and value-pretraining data are also
not fully specified for this exact Qwen3.5-4B/environment combination.

The strongest public reproduction located in this review says explicitly that
it is not author code. It used two nodes and 16 H100s for roughly 500 steps on
24k-token math responses and reported only a directional, statistically
non-significant SAO-over-GRPO+DIS gain. Another implementation labels itself
unofficial and assumes a 3×8-GPU AReaL cluster. These are valuable research
references, not a reliable single-A40 implementation for this project. Do not
implement SAO from the paper summary or copy isolated DIS/GAE snippets and call
the result SAO. Revisit only if the authors release code or a public
implementation demonstrates the full recipe on one 48 GB GPU with preserved
rollout log-probabilities and a validated critic.

The unrelated **Self-Alignment Optimization** generates and self-ranks synthetic
alignment prompts/responses for preference optimization. It has public code,
but it is not arXiv:2607.07508 and does not implement the asynchronous agentic
RL method requested here.

## Smallest informative matrix and parity rules

Use one LoRA per trained arm with the same base revision, bf16/no quantization,
rank 16/alpha 16, target modules, prompt, parser, generation settings, 16
training seeds, development split, and final held-out split. The base arm has no
training budget by definition; all three trained arms receive the same maximum
budget.

First generate one immutable SFT corpus: sample **384 base-Qwen candidate
trajectories** (24 per training seed), retain the two lowest-cost protocol-clean
trajectories per seed, and freeze its manifest before training. Corpus generation
is common preparation overhead, not regenerated differently for each SFT arm.

| Arm | Data/rollout budget | Optimizer budget |
|---|---:|---:|
| Recovered teacher-free RL | 6 updates × 16 seeds × group 4 = **384 complete trajectories** | Up to **13,824 weekly action records** (384 × 36), as already configured |
| Self-generated SFT-only | Frozen shared corpus; save a half-budget branch checkpoint | **6,912** supervised weekly-example presentations to the branch checkpoint, then **6,912** more (13,824 total) |
| Self-generated SFT → RL | Start from the exact SFT-only half-budget checkpoint, then collect **192 on-policy RL trajectories** | **6,912 supervised + 6,912 RL** weekly-example presentations |

This matches trained decision exposures, not just nominal "epochs." The shared
SFT-corpus generation cost is recorded once and divided equally between the two
SFT arms for compute accounting; environment interactions are also reported
separately because offline SFT and on-policy RL do not consume them in the same
way. Impose the same **20 A40-hour ceiling per trained arm** and record actual
GPU-seconds rather than burning unused time. A phase may stop early only under
the common gates below. Do not count cached base-model download time, but do
count allocated candidate generation, training, checkpoint evaluation, and
failed attempts. Use no test seed for data generation, selection, prompt
changes, checkpoint choice, or early stopping.

Evaluate every arm, including base, with deterministic decoding at checkpoint
0, half budget, and full budget on the same frozen development seeds. Report
paired per-seed local cost, mean ± stderr, the exact
0–100 score formula, protocol failure rate, normalized order volatility/bullwhip,
and compute. After selecting and freezing one checkpoint per method from
development results, run validation once; run the new held-out test split once
only after the comparison and reporting code are frozen.

If recovery does not produce a half-budget checkpoint and sufficient timing/
curve evidence, the recovered adapter can be an endpoint reference but not an
equal-schedule controlled arm. A fresh teacher-free RL rerun under the common
schedule is then required for a strict method claim.

## A40 time and cost estimate

Runpod currently lists a Community Cloud A40 at **$0.44/hour**. Recovery billing
for the prior deleted A40 pod provides a conservative end-to-end envelope:
20.6919 GPU-billed hours across 2026-08-02–03, about $9.05 of GPU charge at
approximately $0.4375/hour, and $9.4811 including attached-disk billing
(effective $0.4582/hour). That interval includes setup, smoke, idle time, and the
full run; it is not isolated six-update trainer time. Durable billing evidence
belongs in the recovery record at
`artifacts/live_y_qwen35_4b_rl/recovered/RECOVERY.md`.

Planning estimate, pending timestamps from the recovered log:

- Smoke all three trained methods: at most 2 A40-hours each, **6 hours / about
  $2.64 GPU** total.
- Full SFT-only: likely **4–8 hours / $1.76–$3.52**.
- Full SFT→RL: likely **12–20 hours / $5.28–$8.80**.
- Fresh teacher-free RL rerun, only if required for parity: **12–20 hours /
  $5.28–$8.80**. Clean evaluation of the recovered adapter is much cheaper than
  retraining it.
- Conservative full four-arm plan: **25–40 new A40-hours / $11.00–$17.60 GPU**
  if the recovered RL arm is only re-evaluated, or **40–60 hours /
  $17.60–$26.40** if a fresh teacher-free RL rerun is needed for parity. Add
  roughly 5–10% for persistent disk and operational overhead. These are ceilings,
  not commitments.

## Smoke and expansion gates

1. **Static gate:** exact base/model/tokenizer hashes; split-disjointness tests;
   36-week-plus-settlement replay; 0–128 parser parity; no frozen-spec changes;
   config and seed manifest written before GPU launch.
2. **One-update smoke:** one designated training seed, the smallest viable group
   or SFT batch, at most two A40-hours per method. Require 36 accepted actions,
   100% protocol-clean output, finite loss/advantages, no OOM/NaN, nonzero
   trainable gradient, adapter save/reload, deterministic replay, and complete
   trace/config/checksum output.
3. **Expansion gate:** on the common development split, require two consecutive
   scheduled checkpoints with lower paired mean cost than checkpoint 0, no
   protocol regression, and no order-cap collapse. Treat this only as a go/no-go
   signal, not a held-out claim. Stop after two repeated failures in the same
   phase.
4. **Final gate:** enforce the common exposure and time ceilings; select using
   development only; freeze adapter/config/hash; then validation once. The held-
   out test remains sealed until every method and analysis rule is fixed.

No new training should start until artifact recovery is complete and the
recovered adapter passes a clean, same-split evaluation.

## Primary sources and implementation evidence

- Hou et al., [*Single-Rollout Asynchronous Optimization for Agentic
  Reinforcement Learning*](https://arxiv.org/html/2607.07508), especially the
  method, experimental setup, online-learning simulation, and limitations.
- Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
  Open Language Models*](https://arxiv.org/abs/2402.03300), the original GRPO
  source.
- Qwen, [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), for the
  exact 4B backbone and hybrid architecture.
- [CohenQU/SAO](https://github.com/CohenQU/SAO), an independent reproduction
  that explicitly states the authors released no code/data/checkpoints and
  documents its 16-H100 experiment.
- [fooSynaptic/Single-rollout-async-Optimization](https://github.com/fooSynaptic/Single-rollout-async-Optimization),
  an explicitly unofficial AReaL-based implementation.
- Yin et al., [*Self-Alignment Optimization for Language Models*](https://openreview.net/forum?id=QWMgTMRUnB)
  and its later [public code](https://github.com/SJY8460/SAO), the distinct
  method that shares the SAO acronym.
- Runpod, [GPU Cloud pricing](https://www.runpod.io/pricing), for the current
  public A40 hourly rate.
