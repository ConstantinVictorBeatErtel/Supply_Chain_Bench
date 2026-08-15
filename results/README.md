# Results

The generated [`leaderboard.md`](leaderboard.md) is the public
SupplyChainBench v1.0.0 board. Its `standard` suite freezes the capacity-400
live-Y protocol and the 16-seed manifest. Migrated records retain their source
artifact, commit, timestamp, and publication status; incomplete or
protocol-failed records remain visible but unranked.

Regenerate it after result changes with:

```bash
python -m supplychainbench.leaderboard --include-untracked
```

The command scans tracked result files by default. `--include-untracked` is
useful while preparing a change locally. Generation validates every scanned
JSON file before replacing any leaderboard output.

## Preserved legacy track

The older 20-week serial Beer Game benchmark remains available in
`baseline.json`, with its frozen seeds in `eval/held_out_seeds.json`. It uses
100 demand sequences, a separate score, and a different protocol; it is not
combined with the SupplyChainBench standard board. The historical records are
retained for reproducibility rather than treated as v1 leaderboard entries.

The Verifiers Hub capacity-22 protocol, browser game, and research GRPO
artifacts are likewise preserved in their existing locations. No legacy result
is silently promoted into the frozen standard suite.
