"""OpenRouter chat-completions client for relative-Δ beer-game orders."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from beer_distribution_rl.agents.llm.decode import DecodeResult, ParseFailStats
from beer_distribution_rl.agents.llm.grammar import (
    DEFAULT_DELTA_MAX,
    DEFAULT_ORDER_CAP,
    map_delta_to_order,
)
from beer_distribution_rl.agents.llm.parser import parse_delta_json


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle (fixes macOS Python CERTIFICATE_VERIFY_FAILED)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Prefer flash/lite models first — lower latency for turn-based play.
DEFAULT_OPENROUTER_MODELS: tuple[dict[str, str], ...] = (
    {
        "id": "google/gemini-3.5-flash-lite",
        "label": "Gemini 3.5 Flash Lite (fast)",
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "label": "DeepSeek V4 Flash (fast)",
    },
    {
        "id": "qwen/qwen3.6-flash",
        "label": "Qwen3.6 Flash (fast)",
    },
    {
        "id": "mistralai/ministral-8b-2512",
        "label": "Ministral 8B",
    },
    {
        "id": "qwen/qwen3.5-9b",
        "label": "Qwen3.5 9B",
    },
    {
        "id": "mistralai/mistral-small-2603",
        "label": "Mistral Small",
    },
)


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot be used (missing key / HTTP failure)."""


def openrouter_api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return key or None


def require_openrouter_api_key() -> str:
    key = openrouter_api_key()
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Export it before starting the game:\n"
            '  export OPENROUTER_API_KEY="sk-or-..."\n'
            "  python3 scripts/serve_game.py"
        )
    return key


def model_catalog() -> list[dict[str, str]]:
    """Curated picker list (id + human label)."""
    return [dict(m) for m in DEFAULT_OPENROUTER_MODELS]


@dataclass
class OpenRouterOrderDecoder:
    """Sample relative Δ orders via OpenRouter OpenAI-compatible chat API."""

    model: str
    api_key: str | None = None
    delta_max: int = DEFAULT_DELTA_MAX
    order_cap: int = DEFAULT_ORDER_CAP
    max_parse_retries: int = 2
    temperature: float = 0.0
    timeout_s: float = 45.0
    site_url: str = "https://github.com/ConstantinVictorBeatErtel/Supply_Chain_Bench"
    site_name: str = "Beer Distribution Game"

    def __post_init__(self) -> None:
        self.model = str(self.model).strip()
        if not self.model:
            raise OpenRouterError("OpenRouter model id is required")
        self.api_key = (self.api_key or openrouter_api_key() or "").strip() or None
        self.delta_max = int(self.delta_max)
        self.order_cap = int(self.order_cap)
        self.max_parse_retries = int(self.max_parse_retries)
        self.temperature = float(self.temperature)
        self.timeout_s = float(self.timeout_s)
        self.stats = ParseFailStats()

    def _generate(self, prompt: str) -> str:
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 48,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an ordering agent in the Beer Distribution Game. "
                        f'Emit ONLY JSON: {{"delta": <integer in '
                        f"[{-self.delta_max}, {self.delta_max}]>}}. No other text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout_s, context=_ssl_context()
            ) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise OpenRouterError(
                f"OpenRouter HTTP {exc.code} for model {self.model}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            hint = ""
            reason = str(getattr(exc, "reason", exc))
            if "CERTIFICATE_VERIFY_FAILED" in reason or "SSL" in reason:
                hint = (
                    " Fix: pip3 install -U certifi  "
                    "(or run macOS Python's Install Certificates.command)."
                )
            raise OpenRouterError(f"OpenRouter network error: {exc}.{hint}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError(f"OpenRouter empty choices: {data!r}"[:300])
        msg = choices[0].get("message") or {}
        return str(msg.get("content", ""))

    def sample_order(
        self,
        prompt: str,
        last_demand_or_order: int,
        *,
        stats_key: str = "default",
    ) -> DecodeResult:
        raw = ""
        attempts = 0
        for _ in range(self.max_parse_retries):
            attempts += 1
            raw = self._generate(prompt)
            delta = parse_delta_json(raw, delta_max=self.delta_max)
            if delta is None:
                self.stats.record(key=stats_key, failed=True)
                continue
            order = map_delta_to_order(
                delta,
                last_demand_or_order,
                order_cap=self.order_cap,
                delta_max=self.delta_max,
            )
            self.stats.record(key=stats_key, failed=False)
            return DecodeResult(
                order=order,
                delta=delta,
                raw=raw,
                parse_ok=True,
                n_attempts=attempts,
                used_fallback=False,
                constrained=True,
            )
        fallback = max(0, min(self.order_cap, int(last_demand_or_order)))
        return DecodeResult(
            order=fallback,
            delta=0,
            raw=raw,
            parse_ok=False,
            n_attempts=attempts,
            used_fallback=True,
            constrained=True,
        )
