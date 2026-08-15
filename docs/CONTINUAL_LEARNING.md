# Continual-learning track

## Hypothesis

An agent that accumulates experience across episodes should adapt to a stable
but hidden supply-chain world faster than an agent reset to its base state after
every episode. The test is whether that improvement transfers to fresh demand
seeds without exposing the hidden parameters or test outcomes.

## Modes

| Mode | Weights | Persistent notebook | Test-time updates |
| --- | --- | --- | --- |
| RESET | Frozen | None | None |
| MEMORY | Frozen | One bounded model-authored rewrite | None |
| LEARN | LoRA adapter updates between batches | None | Adapter frozen |

`MEMORY` is not parameter learning: the model writes a bounded JSON notebook
after each adaptation episode from the observations, actions, and visible local
costs it actually saw. The notebook is rewritten, not an unbounded transcript;
the default limit is 4096 UTF-8 bytes and the result records bytes, token count
when available, successful writes, and write failures. Invalid or oversized
writes preserve the previous notebook.

`LEARN` uses the existing token-level GRPO/LoRA machinery in periodic batches.
The adapter and optimizer state persist during adaptation; the final adapter is
frozen before held-out evaluation. Real LEARN runs require a local HF model,
PEFT, and CUDA. `--dry-run` validates the trainer contract without loading a
model or spending on an API.

## Leakage prevention

Each experiment creates one fixed hidden world and separate deterministic
adaptation/test seed manifests. Demand streams vary across episodes while the
hidden dynamics persist. Test seeds are never used to update the notebook,
adapter, prompt, checkpoint choice, or hyperparameters. Test metrics are only
computed after the adaptation state is frozen.

The evaluator never passes the world configuration, event schedule, reference
actions, or final test outcome to the provider. It records those values only in
the evaluator-side result for auditing.

## Metrics and interpretation

Longitudinal output includes episode cost, normalized episode score, rolling
mean, protocol failures, and the suite adaptation/recovery metrics described in
[`BENCHMARK.md`](BENCHMARK.md). LEARN also records optimizer batch boundaries
and total grouped rollouts, because GRPO uses additional common-random-number
replicates per adaptation seed.

An improving MEMORY curve indicates useful experience selection and recall. An
improving LEARN curve indicates an in-weight adapter update under the chosen
training recipe. Neither curve alone establishes general reasoning: compare
against RESET, inspect held-out performance, repeat experiment seeds, and keep
the standard benchmark separate.

Commands:

```bash
python -m supplychainbench.continual --model hf:Qwen/Qwen3.5-4B --mode reset --episodes 100 --output results/continual/reset.json
python -m supplychainbench.continual --model openrouter:anthropic/claude-sonnet-4 --mode memory --episodes 100 --output results/continual/memory.json
python -m supplychainbench.continual --model hf:Qwen/Qwen3.5-4B --mode learn --episodes 100 --output results/continual/learn.json
python -m supplychainbench.plot_continual --reset results/continual/reset.json --memory results/continual/memory.json --learn results/continual/learn.json
```
