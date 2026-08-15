"""RESET/MEMORY/LEARN continual-learning experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from supplychainbench import BENCHMARK_ID, BENCHMARK_VERSION
from supplychainbench.providers import ActionProvider, ProviderError, create_provider, model_slug
from supplychainbench.results import now_utc, write_atomic
from supplychainbench.suites import build_episode, continual_episode_jobs, prompt_for, reference_cost

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MemoryState:
    text: str = ""
    updates: int = 0
    failures: int = 0

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def token_count(self) -> int:
        # Providers may not expose a tokenizer. This stable fallback is still
        # useful for enforcing/reporting the notebook contract; HF providers
        # can replace it with tokenizer-specific accounting at the call site.
        return len(self.text.split()) if self.text else 0


def _memory_prompt_summary(job: Any, episode: Any, actions: list[int], weekly: list[float]) -> str:
    return json.dumps({"suite": job.suite, "seed": job.seed, "actions": actions,
                       "weekly_local_costs": weekly,
                       "completed_weeks": len(actions),
                       "protocol_clean": bool(episode.outcome and episode.outcome["grade"].get("protocol_clean"))},
                      sort_keys=True, separators=(",", ":"))


def _run_one(provider: ActionProvider, job: Any, notebook: str = "", *, allow_memory_update: bool = False,
             memory_limit_bytes: int = 4096, memory_state: MemoryState | None = None) -> dict[str, Any]:
    episode = build_episode(job)
    observation = episode.start()
    provider.reset_episode(job, episode, observation)
    actions: list[int] = []
    weekly: list[float] = []
    raw: list[str] = []
    failure: str | None = None
    while not episode.done:
        system, user = prompt_for(job, observation)
        if notebook:
            system += "\nPersistent notebook from earlier episodes (may be incomplete):\n" + notebook
        result = provider.act(system, user, observation)
        raw.append(result.raw)
        if result.quantity is None:
            failure = result.error or "invalid_model_action"
            episode.protocol_failure_outcome(error_count=1, category=failure)
            break
        action = int(result.quantity)
        outcome = episode.place_order(action)
        actions.append(action)
        weekly.append(float(episode.operational_transitions[-1]["local_costs"]["wholesaler"]))
        if not outcome["done"]:
            observation = outcome["next_observation"]
    grade = episode.outcome["grade"] if episode.outcome else {}
    clean = bool(grade.get("protocol_clean", False))
    local = grade.get("primary", {}).get("local_total_cost") if clean else None
    ref = reference_cost(job)
    memory_error = None
    if allow_memory_update and memory_state is not None:
        memory_result = provider.write_memory(
            "You maintain a factual notebook for a supply-chain decision agent. Never invent hidden state.",
            _memory_prompt_summary(job, episode, actions, weekly), memory_state.text, memory_limit_bytes,
        )
        candidate = memory_result.raw if memory_result.error is None else ""
        if candidate and len(candidate.encode("utf-8")) <= memory_limit_bytes:
            memory_state.text = candidate
            memory_state.updates += 1
        else:
            memory_state.failures += 1
            memory_error = memory_result.error or "memory exceeded configured byte limit"
    return {"phase": "adaptation", "seed": job.seed, "index": job.index,
            "protocol_clean": clean, "failure": failure,
            "local_total_cost": float(local) if local is not None else None,
            "reference_cost": float(ref["local_total_cost"]),
            "normalized_score": 100.0 * float(ref["local_total_cost"]) / float(local) if local else None,
            "actions": actions, "weekly_local_costs": weekly,
            "memory_bytes": memory_state.bytes if memory_state else 0,
            "memory_token_count": memory_state.token_count if memory_state else 0,
            "memory_update_error": memory_error, "raw_outputs": raw}


class ContinualTrainer:
    """Interface for parameter-updating continual trainers."""

    def adapt(self, jobs: list[Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def freeze(self) -> None:
        pass


class GRPOBatchTrainer(ContinualTrainer):
    """Adapter over the repository's existing token-level GRPO update loop."""

    def __init__(self, model_name: str, *, adapter: str | None, group_size: int, batch_episodes: int, output_dir: Path, dry_run: bool = False):
        self.model_name = model_name
        self.adapter = adapter
        self.group_size = group_size
        self.batch_episodes = batch_episodes
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.rollout_count = 0
        self.optimizer_update_boundaries: list[dict[str, int]] = []
        if not model_name.startswith("hf:"):
            raise ProviderError("LEARN requires an hf:<model> provider")
        self.model_identifier = model_name.split(":", 1)[1]

    def adapt(self, jobs: list[Any]) -> list[dict[str, Any]]:
        if self.dry_run:
            self.rollout_count = len(jobs) * self.group_size
            for start in range(0, len(jobs), self.batch_episodes):
                batch = jobs[start:start + self.batch_episodes]
                self.optimizer_update_boundaries.append({"batch_index": start // self.batch_episodes,
                                                         "episode_start": start,
                                                         "episode_end": start + len(batch),
                                                         "rollouts": len(batch) * self.group_size})
            return [{"phase": "adaptation", "seed": job.seed, "trainer": "grpo", "dry_run": True,
                     "rollout_count": self.group_size, "optimizer_update": job.index // self.batch_episodes}
                    for job in jobs]
        try:
            import torch
            import scripts.train_colab_grpo_wholesaler as pilot
        except ImportError as exc:
            raise ProviderError("LEARN requires torch and the existing GRPO trainer dependencies") from exc
        if not torch.cuda.is_available():
            raise ProviderError("LEARN requires CUDA; use --dry-run to validate the trainer on CPU")
        args = SimpleNamespace(model_name=self.model_identifier, adapter=self.adapter, eval_only=False,
                               no_4bit=True, output_dir=str(self.output_dir), prompt_max_tokens=2048,
                               max_new_tokens=32, temperature=0.7, top_p=0.95, train_minibatch=4,
                               inference_minibatch=4, learning_rate=1e-5, clip_epsilon=0.2,
                               dual_clip=3.0, all_completion_tokens=False, _prefix_kv_cache=None)
        model, tokenizer = pilot.load_policy(args)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        original_start, original_prompt, original_extract = pilot.start_episode, pilot.prompt_text, pilot.extract_quantity
        output_rows: list[dict[str, Any]] = []
        try:
            def _start(task: Any, group_id: int) -> Any:
                episode = build_episode(task.data.job)
                return pilot.EpisodeRun(group_id, task.data.name, episode, episode.start())
            pilot.start_episode = _start
            def _prompt(task: Any, observation: dict[str, Any], tok: Any) -> str:
                system, user = prompt_for(task.data.job, observation)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                try:
                    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pilot.prompt_text = _prompt
            from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion
            pilot.extract_quantity = lambda text: parse_completion(text, tokenizer=tokenizer)
            for start in range(0, len(jobs), self.batch_episodes):
                batch = jobs[start:start + self.batch_episodes]
                self.rollout_count += len(batch) * self.group_size
                self.optimizer_update_boundaries.append({"batch_index": start // self.batch_episodes,
                                                         "episode_start": start,
                                                         "episode_end": start + len(batch),
                                                         "rollouts": len(batch) * self.group_size})
                tasks = [SimpleNamespace(data=SimpleNamespace(name=f"scb:{job.suite}:{job.seed}", job=job, scenario=job.spec.to_dict(), controlled_role="wholesaler")) for job in batch]
                runs = pilot.rollout_batch(model, tokenizer, tasks, args, self.group_size, sample=True)
                pilot.assign_advantages(runs)
                stats = pilot.train_update(model, optimizer, [record for run in runs for record in run.records], args)
                for run in runs:
                    row = pilot.episode_summary(run)
                    row.update({"phase": "adaptation", "seed": run.task_name.rsplit(":", 1)[-1],
                                "trainer_stats": stats, "rollout_count": self.group_size,
                                "optimizer_update": start // self.batch_episodes})
                    output_rows.append(row)
                adapter_dir = self.output_dir / f"batch_{start // self.batch_episodes:03d}"
                model.save_pretrained(adapter_dir)
            self.adapter = str(self.output_dir / f"batch_{max(0, (len(jobs) - 1) // self.batch_episodes):03d}")
            return output_rows
        finally:
            pilot.start_episode, pilot.prompt_text, pilot.extract_quantity = original_start, original_prompt, original_extract

    def freeze(self) -> None:
        if self.adapter is None:
            raise ProviderError("LEARN completed without an adapter checkpoint")


def run(model: str, mode: str, *, suite: str = "held_out_dynamics", episodes: int = 100,
        test_episodes: int = 16, output: Path | None = None, adapter: str | None = None,
        memory_limit_bytes: int = 4096, group_size: int = 4, batch_episodes: int = 8,
        dry_run: bool = False) -> dict[str, Any]:
    if mode not in {"reset", "memory", "learn"}:
        raise ValueError("mode must be reset, memory, or learn")
    adaptation_jobs = list(continual_episode_jobs(suite, "adaptation", episodes))
    test_jobs = list(continual_episode_jobs(suite, "test", test_episodes))
    output = output or ROOT / "results/continual" / f"{model_slug(model)}-{mode}.json"
    # RESET deliberately constructs a fresh provider/session for every
    # episode. MEMORY keeps one frozen-weight provider but still receives only
    # the bounded notebook, never an implicit transcript.
    provider = None if mode in {"learn", "reset"} else create_provider(model, adapter=adapter)
    memory = MemoryState()
    if mode == "learn":
        trainer = GRPOBatchTrainer(model, adapter=adapter, group_size=group_size, batch_episodes=batch_episodes, output_dir=output.parent / f"{model_slug(model)}-adapter", dry_run=dry_run)
        adaptation_rows = trainer.adapt(adaptation_jobs)
        trainer.freeze() if not dry_run else None
        provider = create_provider(model, adapter=trainer.adapter if not dry_run else adapter) if not dry_run else create_provider("agent:constant-18")
    else:
        adaptation_rows = []
        for job in adaptation_jobs:
            if mode == "memory":
                # MEMORY receives the notebook explicitly, while provider
                # weights remain fixed across the adaptation stream.
                adaptation_rows.append(_run_one(provider, job, memory.text,
                                                allow_memory_update=True,
                                                memory_limit_bytes=memory_limit_bytes,
                                                memory_state=memory))
                continue
            episode_provider = create_provider(model, adapter=adapter)
            try:
                adaptation_rows.append(_run_one(episode_provider, job, "", allow_memory_update=False,
                                                memory_limit_bytes=memory_limit_bytes, memory_state=memory))
            finally:
                episode_provider.close()
    frozen_memory = memory.text if mode == "memory" else ""
    test_rows = []
    for job in test_jobs:
        episode_provider = create_provider(model, adapter=adapter) if mode == "reset" else provider
        try:
            test_rows.append(_run_one(episode_provider, job, frozen_memory if mode == "memory" else "",
                                      allow_memory_update=False, memory_limit_bytes=memory_limit_bytes,
                                      memory_state=memory))
        finally:
            if mode == "reset":
                episode_provider.close()
        test_rows[-1]["phase"] = "test"
    if provider is not None:
        provider.close()
    payload = {"schema_version": "1.0.0", "benchmark": {"id": BENCHMARK_ID, "version": BENCHMARK_VERSION},
               "model": model, "mode": mode, "suite": suite,
               "configuration": {"episodes": episodes, "test_episodes": test_episodes, "memory_limit_bytes": memory_limit_bytes, "group_size": group_size, "batch_episodes": batch_episodes},
               "adaptation": adaptation_rows, "test": test_rows,
               "memory": {"final_bytes": memory.bytes, "final_tokens": memory.token_count,
                          "updates": memory.updates, "failures": memory.failures,
                          "frozen_for_test": frozen_memory},
               "learning": {"rollout_count": (trainer.rollout_count if mode == "learn" else 0),
                            "optimizer_update_boundaries": (trainer.optimizer_update_boundaries if mode == "learn" else []),
                            "adapter_frozen": mode == "learn" and not dry_run},
               "run": {"timestamp": now_utc(), "dry_run": dry_run, "test_seeds": [job.seed for job in test_jobs]}}
    write_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RESET/MEMORY/LEARN experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("reset", "memory", "learn"), required=True)
    parser.add_argument("--suite", default="held_out_dynamics")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter")
    parser.add_argument("--memory-limit-bytes", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args.model, args.mode, suite=args.suite, episodes=args.episodes, test_episodes=args.test_episodes,
                      output=args.output, adapter=args.adapter, memory_limit_bytes=args.memory_limit_bytes,
                      group_size=args.group_size, batch_episodes=args.batch_episodes, dry_run=args.dry_run)
    except ProviderError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"mode": payload["mode"], "adaptation_episodes": len(payload["adaptation"]), "test_episodes": len(payload["test"])}, indent=2))


if __name__ == "__main__":
    main()
