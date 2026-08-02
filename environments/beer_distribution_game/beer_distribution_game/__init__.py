"""Prime Intellect Beer Distribution Game environment.

The simulator modules remain importable without Verifiers so domain tests and
non-Hub baselines do not acquire a framework dependency. The Hub package itself
declares Verifiers 0.2.0, so its native exports are always present when installed.
"""

__version__ = "0.2.2"

try:
    from .harness import BeerHarness, BeerHarnessConfig
    from .taskset import BeerTaskset, BeerTasksetConfig
except ModuleNotFoundError as exc:
    if exc.name != "verifiers":
        raise
    __all__: list[str] = []
else:
    import verifiers.v1 as vf

    def load_taskset(config: BeerTasksetConfig) -> BeerTaskset:
        """Prime package loader for the Beer Game taskset."""
        return BeerTaskset(config=config)


    def load_harness(config: BeerHarnessConfig) -> BeerHarness:
        """Prime package loader for the Beer Game harness."""
        return BeerHarness(config=config)


    def load_environment(**kwargs: object):
        """Prime Environments Hub entry point.

        Verifiers 0.2 loads published packages through its legacy multi-turn
        interface.  The adapter retains the native weekly ``place_order`` tool
        protocol while the v1 taskset/harness remain available to direct users.
        """
        from .legacy_env import BeerGameEnv

        return BeerGameEnv(**kwargs)


    __all__ = ["BeerTaskset", "BeerHarness", "load_taskset", "load_harness", "load_environment"]
