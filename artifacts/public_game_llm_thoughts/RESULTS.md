# Public game comparison traces, with a per-week rationale

`traces.json` is the recorded opponent the static web game plays you against, and
the source of the debrief's week-by-week thought tracker. It replaces the
DeepSeek V4 Flash Hub traces that previously fed
`static_web/scripts/build.js`.

**Why a new run.** The Hub traces store actions only. Their eval protocol runs
with reasoning disabled and a strict `{"quantity": n}` reply, so nothing the
model *thought* survives anywhere in this repository. A tracker showing weekly
reasoning needs a run where the rationale and the order come out of the same
forward pass — reconstructing a rationale after the fact would be fabrication,
not a record.

## Setup

| Field | Value |
|---|---|
| Model | `x-ai/grok-4.5` via OpenRouter — the top scorer on the capacity-400 board |
| Scenario | `t5-strategic-y-v2`, Tier-5 headline Y, wholesaler seat |
| Seeds | The eight playable seeds: development 0–2, validation 0–4 |
| Factory capacity | **400** — the public play capacity, not the research value of 22 |
| Horizon | 36 operational weeks plus native settlement |
| Protocol | One JSON object per week: `{"thought": ..., "quantity": ...}` |
| Decoding | temperature 0.7, top-p 0.95, max_tokens 768, `reasoning.effort=low` |
| Cost | $1.18 for 295 calls (299,528 prompt / 118,826 completion tokens) |

Capacity 400 matters: the recorded model now plays *the same game the human
plays*, under the same disclosed capacity. The previous traces were produced
under a capacity-22 prompt and then replayed at 400, so the model's stated
beliefs and the player's board disagreed. That inconsistency would have become
visible the moment the notes were put on screen.

The system prompt is the Hub prompt with its capacity sentence corrected
(`beer_distribution_game.prompts.public_demand_blurb` hard-codes 22, which is a
false statement in a capacity-400 episode), plus the reply-format instruction.

## Result

All eight episodes are protocol-clean and all 288 weeks carry a note.

| Seed | Model cost | Base-stock | Ratio | Format retries |
|---|---:|---:|---:|---:|
| development-0 | 3,960.0 | 655.0 | 6.0x | 1 |
| development-1 | 985.5 | 410.0 | 2.4x | 0 |
| development-2 | 2,126.0 | 772.0 | 2.8x | 0 |
| validation-0 | 5,437.0 | 600.0 | 9.1x | 4 |
| validation-1 | 2,488.5 | 669.0 | 3.7x | 1 |
| validation-2 | 2,718.5 | 622.5 | 4.4x | 1 |
| validation-3 | 2,701.5 | 898.5 | 3.0x | 0 |
| validation-4 | 1,198.0 | 285.0 | 4.2x | 0 |
| **Mean** | **2,701.9** | **614.0** | **4.4x** | 7 |

Format retries are weeks where the first reply came back empty — hidden
reasoning consumed the completion budget — and a retry produced a valid order.
No episode ever exhausted its retries.

**This is a weaker opponent than the traces it replaces.** Replaying the old
DeepSeek actions at capacity 400 gives a mean local cost of 1,646.9. Grok's own
notes explain why it does worse: told the factory can make 400 a week, it orders
45 → 120 → 128 against a 19-unit backlog in weeks 2–5, then spends twenty weeks
burning off the inventory that arrives. The old traces ordered a near-constant
22 because they were produced under a capacity-22 prompt, and that accidental
constancy is a decent wholesaler policy. The new number is the honest cost of a
frontier model playing the actual public game.

## Control: does asking for a rationale change the play?

No — it helped, if anything. `control_json_only_development_0.json` re-runs
development-0 with the same model, seed, and capacity but the published
JSON-only reply format:

| Protocol | development-0 local cost |
|---|---:|
| With per-week rationale (shipped) | 3,960.0 |
| JSON-only control | 8,293.0 |

Reproduce with `scripts/check_public_game_thought_protocol_control.py`. One
episode at temperature 0.7 is not a significance test, but it rules out the
concern that the tracker's notes come from a handicapped run.

## Reproducing

```bash
OPENROUTER_API_KEY=sk-or-... python scripts/run_public_game_llm_thoughts.py
npm run test        # rebuilds the catalog and re-verifies every trace
```

`static_web/test/trace_integrity.test.js` replays each recorded action sequence
in the JavaScript simulator at this capacity and asserts the published
`episode_id`, costs, and reward; `parity.test.js` checks the same replays
against the frozen Python environment.
