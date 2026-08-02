# Frozen wholesaler benchmark split

`held_out_seeds.json` is the immutable 100-episode evaluation split for the
20-week serial Beer Distribution Game benchmark. The seeds were drawn once
from `random.Random(20260802)` and committed before policy evaluation,
training, or data generation.

The benchmark controller is the wholesaler. Retailer, distributor, and factory
remain scripted base-stock policies. Training and data-generation code must
reject any seed listed in `held_out_seeds.json`.

This file must never be regenerated or edited in place. Create a new benchmark
version and a new split file for a future evaluation protocol.
