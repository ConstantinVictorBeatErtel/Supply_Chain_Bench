# Tier-5 Y Qwen3.5-4B LoRA training

This note documents the completed teacher-free training run that produced the
development result in [`artifacts/live_y_qwen35_4b_rl/`](../artifacts/live_y_qwen35_4b_rl/).
It does not modify the frozen environment, reward, or difficulty contracts.

## Task and data separation

The model controlled the wholesaler in the native `v0.2.0` Tier-5 Y-headline
scenario for 36 operational weeks followed by the native three-week settlement.
The other roles used the environment's scripted policies, including the
strategic retailer claimants. The model saw only the role-local observation and
had to emit one integer order from 0 to 128 per week.

`experiments/live_y_qwen_rl/splits.json` freezes 16 training and 10 evaluation
seed IDs. They are derived as
`SHA-256('beer-live-y-rl-v1|<split>|<index>')[:16]`, are mutually disjoint, and
are expressly separate from the public live-game and normative benchmark splits.

## Method

- Base model: `Qwen/Qwen3.5-4B` in bf16; no 4-bit quantization.
- Adapter: LoRA rank 16, alpha 16, no dropout, targeting `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Optimization: six updates, group size four, training minibatch four,
  learning rate `5e-6`, training seed `20260806`.
- Objective: group-relative advantage from **negative terminal local wholesaler
  cost**. A protocol failure receives `-100000`.
- No teacher: the LoRA began from the base model without demonstrations,
  distillation targets, or base-stock actions. The base-stock policy was used
  only as an evaluation reference.
- Inference: plain Transformers generation, maximum eight generated tokens,
  temperature `0.7`, top-p `0.95`; no vLLM.

The run was executed on one cloud-hosted NVIDIA A40 (48 GB VRAM) with artifacts
kept under Runpod's persistent `/workspace` volume. The adapter, pre/post
evaluation JSON, rollout trace, config, metrics, and log remain in
`/workspace/outputs/beer-wholesaler-qwen35-4b-live-y-rl/`.

## Reproduce

On a CUDA-capable host with the repository installed and `PYTHONPATH` including
the Hub package:

```bash
PYTHONPATH=.:environments/beer_distribution_game \
python scripts/train_qwen35_live_y_rl.py \
  --updates 6 \
  --group-size 4 \
  --train-minibatch 4 \
  --output-dir /workspace/outputs/beer-wholesaler-qwen35-4b-live-y-rl
```

The executed source revision on Runpod was `221462a2309a583144962a514a65159700fdef34`.
See the run summary for the reported pre/post result and scoring calculation.
