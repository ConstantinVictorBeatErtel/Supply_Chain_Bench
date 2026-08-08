"""Independent PPO (IPPO) — one policy and critic per role.

Heavy training deps (numpy/torch) live in ``trainer`` / ``matrix``. Keep this
package init lazy so light callers (playable web UI, LLM unit tests) can import
submodules like ``policy_agent`` without requiring the marl extra.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "IPPOConfig",
    "IPPOTrainer",
    "MatrixCell",
    "build_env_config",
    "capacity_tag",
    "default_run_name",
    "enumerate_cells",
    "git_sha",
    "prune_summary",
]

_TRAINER_EXPORTS = frozenset(
    {
        "IPPOConfig",
        "IPPOTrainer",
        "build_env_config",
        "capacity_tag",
        "default_run_name",
        "git_sha",
    }
)
_MATRIX_EXPORTS = frozenset({"MatrixCell", "enumerate_cells", "prune_summary"})


def __getattr__(name: str) -> Any:
    if name in _TRAINER_EXPORTS:
        from beer_distribution_rl.agents.ippo import trainer as _trainer

        return getattr(_trainer, name)
    if name in _MATRIX_EXPORTS:
        from beer_distribution_rl.agents.ippo import matrix as _matrix

        return getattr(_matrix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
