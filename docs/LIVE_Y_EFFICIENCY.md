# Live-Y Qwen3.5-4B efficiency contract

This document records the engineering choices that reduce compute while
keeping the experiment interpretable and information-parity compliant.

## What is cached

`use_cache=True` during generation is necessary but only reuses keys and
values between newly generated tokens inside one `generate()` call. It does
not reuse the prompt on the following week. The trainer now adds
`PrefixKVCache`: it tokenizes the current batch, finds the longest shared
prefix, and caches only that invariant prefix. In this experiment that is
normally the system message and chat-template prefix. The observation and
rolling history remain uncached and are recomputed every week, as they must.

The cache is invalidated after every optimizer update because LoRA changes the
model weights. Cache API failures or short prefixes fall back to the original
full-prompt batched generation path. `training_forward_use_cache=False` is
explicit: KV caching is useful for autoregressive decoding, not for the
teacher-forced backward pass, where it increases memory use.

Per-update `training_metrics.json` records prompt tokens, completion tokens,
the generation cap, cache hits, prefix forwards, fallbacks, and estimated
prefix tokens avoided. A fallback counter is a correctness signal, not a
reason to silently trust an unsupported cache implementation.

## Cost controls already in use

- The research prompt exposes only the wholesaler's local observation and the
  bounded rolling history; demand parameters, the factory capacity, and other
  roles' private information are omitted.
- The model is asked for one strict JSON quantity, with thinking disabled at
  chat-template application and a bounded completion. Training defaults to a
  32-token generation cap; the protocol's 192-token parser limit remains an
  upper validity contract, not a reason to decode 192 tokens on every action.
- Weekly prompts are rebuilt from a bounded recent-history window rather than
  an ever-growing transcript.
- Active trajectories are generated in batches, and action log-probability
  and training passes use separate minibatches.
- The model is loaded once per run and trained with LoRA, bf16 weights, and
  gradient checkpointing. The backward pass uses `use_cache=False` and
  `set_to_none=True` optimizer gradients.
- `torch.inference_mode()` is used for generation and old-policy scoring;
  allocator cleanup happens before the peak-memory backward phase.
- Training uses local role cost, same-timestep group baselines, fixed
  deterministic seed derivation, and common random numbers. This reduces
  variance and avoids spending additional rollouts on a learned critic.
- Training and evaluation seed registries are separate, and reference
  trajectories are not constructed for training rollouts.

## Additional ideas considered

The following are intentionally not enabled by default:

- FlashAttention/SDPA should be benchmarked on the exact Runpod image. It is
  a worthwhile next hardware optimization, but silently changing kernels can
  confound the small research comparison.
- `torch.compile` is not enabled because dynamic weekly sequence lengths,
  PEFT wrappers, and Qwen custom generation code can trigger recompiles or
  graph breaks that cost more than they save for this workload.
- vLLM and quantized serving are not used in the training path. They may be
  excellent for large evaluation sweeps, but would change the single-model
  rollout/update semantics and the current bf16/no-4bit experiment contract.
- An 8-bit optimizer could reduce optimizer memory, but LoRA already makes
  optimizer state small relative to the frozen 4B base model; it should be a
  separate ablation, not an untracked change.
- Full cross-week KV reuse is invalid because the observation and history
  change after every action. Reusing those tensors would give the model stale
  information. Prefix-only reuse is the maximum safe reuse without changing
  the environment protocol.

For future scale-up, benchmark SDPA/FlashAttention and length-bucketed
teacher-forcing batches, then keep only changes that improve measured
tokens-per-second or peak VRAM without changing prompt visibility or seed
semantics.
