"""Compatibility environment for the Verifiers 0.2 Hub loader.

The domain taskset and harness use the native v1 API.  Prime's package-quality
runner in Verifiers 0.2 loads a legacy ``MultiTurnEnv`` factory, so this adapter
exposes the same weekly ``place_order`` interaction through that stable surface.
"""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset
import verifiers as vf

from .episode import BeerEpisode
from .prompts import system_prompt
from .scenario import scenario_for


_ROLE = "wholesaler"
_TOOL_DEF = {
    "name": "place_order",
    "description": "Place this week's replenishment order as an integer from 0 through 128.",
    "parameters": {
        "type": "object",
        "properties": {"quantity": {"type": "integer", "minimum": 0, "maximum": 128}},
        "required": ["quantity"],
        "additionalProperties": False,
    },
}


class BeerGameEnv(vf.MultiTurnEnv):
    """A rolling, tool-call Beer Game episode for the legacy Hub evaluator."""

    def __init__(self, max_turns: int = 38, **kwargs: Any) -> None:
        spec = scenario_for(5, "development", 0)
        self._spec = spec
        preview = BeerEpisode(spec, _ROLE, include_reference=False)
        initial_observation = preview.start()
        prompt = "Current observation:\n" + json.dumps(
            initial_observation, sort_keys=True, separators=(",", ":")
        )
        dataset = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "scenario": spec.to_dict(),
                    "controlled_role": _ROLE,
                }
            ]
        )
        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            system_prompt=system_prompt(spec, _ROLE),
            tool_defs=[_TOOL_DEF],
            max_turns=max_turns,
            **kwargs,
        )

    async def setup_state(self, state: vf.State) -> vf.State:
        # The legacy Hub worker retains the standardized prompt payload but can
        # drop custom dataset columns.  This adapter serves one immutable
        # scenario, so keep its canonical spec on the environment instance.
        episode = BeerEpisode(self._spec, _ROLE, include_reference=False)
        episode.start()
        state["beer_episode"] = episode
        state["beer_outcome"] = None
        state["protocol_errors"] = 0
        return state

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> vf.Messages:
        del kwargs
        last_message = messages[-1]
        calls = getattr(last_message, "tool_calls", None) or []
        if len(calls) != 1 or getattr(calls[0], "name", None) != "place_order":
            return self._protocol_error(state, calls, "Call place_order exactly once.")

        call = calls[0]
        try:
            arguments = json.loads(call.arguments)
            quantity = arguments["quantity"]
            if type(quantity) is not int or not 0 <= quantity <= 128 or set(arguments) != {"quantity"}:
                raise ValueError("quantity must be an integer from 0 through 128")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return self._protocol_error(state, calls, "quantity must be an integer from 0 through 128")

        result = state["beer_episode"].place_order(quantity)
        message = vf.ToolMessage(
            role="tool", content=json.dumps(result, sort_keys=True), tool_call_id=call.id
        )
        if result["done"]:
            state["beer_outcome"] = state["beer_episode"].outcome
            state["final_env_response"] = [message]
        return [message]

    def _protocol_error(self, state: vf.State, calls: list[Any], detail: str) -> vf.Messages:
        state["protocol_errors"] += 1
        tool_call_id = getattr(calls[0], "id", "protocol_error") if calls else "protocol_error"
        payload = {
            "status": "error",
            "message": detail,
            "protocol_error_count": state["protocol_errors"],
        }
        message = vf.ToolMessage(role="tool", content=json.dumps(payload), tool_call_id=tool_call_id)
        if state["protocol_errors"] >= 3:
            state["final_env_response"] = [message]
        return [message]

    @vf.reward
    async def beer_game_reward(self, state: vf.State, **kwargs: Any) -> float:
        del kwargs
        outcome = state.get("beer_outcome")
        if outcome is None:
            return 0.0
        return float(outcome["grade"]["episode_reward"])
