"""LLM text I/O: obs→prompt serializer + grammar-constrained order parser.

Product layer clearing readiness Checks 2–3 (text I/O + rolling context) and
feeding Check 7 budget arithmetic. Order-only agents; broadcast / cheap-talk
channel is out of scope. Action space matches IPPO relative Δ ∈ [-8, 8] →
absolute order clipped to ``order_cap`` (default 128). Rolling own-history
window defaults to ``DEFAULT_ROLLING_WINDOW`` (8).
"""

from __future__ import annotations

from beer_distribution_rl.agents.llm.decode import (
    ConstrainedOrderDecoder,
    DecodeResult,
    ParseFailStats,
)
from beer_distribution_rl.agents.llm.openrouter import (
    OpenRouterError,
    OpenRouterOrderDecoder,
    model_catalog,
    openrouter_api_key,
)
from beer_distribution_rl.agents.llm.role_agent import LLMRoleAgent
from beer_distribution_rl.agents.llm.grammar import (
    DELTA_JSON_SCHEMA,
    ORDER_DELTA_GBNF,
    delta_json_schema,
    map_delta_to_order,
)
from beer_distribution_rl.agents.llm.memory import (
    DEFAULT_ROLLING_WINDOW,
    AgentMemory,
    WeekRecord,
)
from beer_distribution_rl.agents.llm.parser import parse_delta_json, parse_order_legacy
from beer_distribution_rl.agents.llm.serializer import (
    FORBIDDEN_SUBSTRINGS,
    OWN_HISTORY_FIELDS,
    estimate_prompt_tokens,
    observe_local,
    prompt_leak_report,
    serialize_prompt,
)

__all__ = [
    "DEFAULT_ROLLING_WINDOW",
    "AgentMemory",
    "ConstrainedOrderDecoder",
    "DELTA_JSON_SCHEMA",
    "DecodeResult",
    "FORBIDDEN_SUBSTRINGS",
    "LLMRoleAgent",
    "ORDER_DELTA_GBNF",
    "OWN_HISTORY_FIELDS",
    "OpenRouterError",
    "OpenRouterOrderDecoder",
    "ParseFailStats",
    "WeekRecord",
    "delta_json_schema",
    "estimate_prompt_tokens",
    "map_delta_to_order",
    "model_catalog",
    "observe_local",
    "openrouter_api_key",
    "parse_delta_json",
    "parse_order_legacy",
    "prompt_leak_report",
    "serialize_prompt",
]
