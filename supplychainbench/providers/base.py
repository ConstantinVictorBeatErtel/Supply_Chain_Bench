"""Provider abstraction with deterministic local baselines and lazy API/HF adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from supplychainbench.suites import SuiteEpisode


class ProviderError(RuntimeError):
    pass


@dataclass
class ActionResponse:
    quantity: int | None
    raw: str = ""
    error: str | None = None
    usage: dict[str, Any] | None = None


class ActionProvider:
    provider_kind = "base"
    supports_memory = False
    supports_learning = False

    def reset_episode(self, job: SuiteEpisode, episode: Any, observation: dict[str, Any]) -> None:
        del job, episode, observation

    def act(self, system: str, user: str, observation: dict[str, Any]) -> ActionResponse:
        raise NotImplementedError

    def write_memory(self, system: str, user: str, previous: str, limit_bytes: int) -> ActionResponse:
        del system, user, previous, limit_bytes
        return ActionResponse(None, error="provider does not support memory writes")

    def close(self) -> None:
        pass


class BaselineProvider(ActionProvider):
    provider_kind = "agent"

    def __init__(self, policy_id: str):
        self.policy_id = policy_id
        self.policy = None

    def reset_episode(self, job: SuiteEpisode, episode: Any, observation: dict[str, Any]) -> None:
        from beer_distribution_game.policies import adaptive_policy
        self.policy = adaptive_policy(job.spec, "wholesaler") if self.policy_id == "adaptive" else None

    def act(self, system: str, user: str, observation: dict[str, Any]) -> ActionResponse:
        del system, user
        if self.policy_id == "constant-18":
            return ActionResponse(18, raw='{"quantity":18}')
        if self.policy is None:
            return ActionResponse(None, error="baseline was not reset")
        quantity = int(self.policy.act(observation))
        return ActionResponse(quantity, raw=json.dumps({"quantity": quantity}))


def _parse_quantity(raw: str, *, tokenizer: Any = None) -> int | None:
    from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()
    return parse_completion(value, tokenizer=tokenizer)


class _HTTPProvider(ActionProvider):
    provider_kind = "api"
    supports_memory = True

    def __init__(self, model_id: str, *, api_key_env: str, base_url: str, temperature: float = 0.7):
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise ProviderError(f"{api_key_env} is required for model provider {model_id!r}")

    def _complete(self, system: str, user: str, *, memory: bool = False) -> ActionResponse:
        if memory:
            user = user + "\nReturn exactly JSON: {\"memory\": \"bounded summary\"}."
        body = {
            "model": self.model_id,
            "temperature": 0.0 if memory else self.temperature,
            "max_tokens": 256 if memory else 64,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            return ActionResponse(None, error=f"provider request failed: {exc}")
        choices = payload.get("choices") or []
        content = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return ActionResponse(_parse_quantity(str(content)), raw=str(content), usage=payload.get("usage"))

    def act(self, system: str, user: str, observation: dict[str, Any]) -> ActionResponse:
        del observation
        return self._complete(system, user)

    def write_memory(self, system: str, user: str, previous: str, limit_bytes: int) -> ActionResponse:
        prompt = f"Existing notebook:\n{previous or '(empty)'}\n\nVisible episode summary:\n{user}\n\nRewrite the notebook in at most {limit_bytes} UTF-8 bytes."
        result = self._complete(system, prompt, memory=True)
        if result.raw:
            try:
                value = json.loads(result.raw.strip().strip('`'))
                text = value.get("memory") if isinstance(value, dict) else None
                if isinstance(text, str):
                    result.raw = text
            except json.JSONDecodeError:
                result.error = "memory response was not JSON"
        return result


class HFProvider(ActionProvider):
    provider_kind = "hf"
    supports_memory = True
    supports_learning = True

    def __init__(self, model_id: str, *, adapter: str | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderError("HF provider requires the optional hf dependencies (torch, transformers, peft)") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
        self.model.to(device).eval()
        if adapter:
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False).to(device).eval()
            except ImportError as exc:
                raise ProviderError("loading an HF adapter requires peft") from exc
        self.device = device

    def _generate(self, system: str, user: str, max_new_tokens: int = 64) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def act(self, system: str, user: str, observation: dict[str, Any]) -> ActionResponse:
        del observation
        raw = self._generate(system, user)
        return ActionResponse(_parse_quantity(raw, tokenizer=self.tokenizer), raw=raw)

    def write_memory(self, system: str, user: str, previous: str, limit_bytes: int) -> ActionResponse:
        raw = self._generate(system, f"Existing notebook:\n{previous or '(empty)'}\nVisible episode summary:\n{user}\nRewrite in <= {limit_bytes} UTF-8 bytes as JSON {{\"memory\":\"...\"}}.", 128)
        try:
            value = json.loads(raw.strip().strip('`'))
            text = value.get("memory") if isinstance(value, dict) else None
            return ActionResponse(None, raw=text or "", error=None if isinstance(text, str) else "invalid memory JSON")
        except json.JSONDecodeError:
            return ActionResponse(None, raw=raw, error="invalid memory JSON")


def model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-").lower()


def create_provider(model: str, *, adapter: str | None = None, base_url: str | None = None, api_key_env: str | None = None) -> ActionProvider:
    kind, sep, identifier = model.partition(":")
    if not sep or not identifier:
        raise ProviderError("model must use a provider URI such as agent:adaptive, openrouter:..., or hf:...")
    if kind == "agent" and identifier in {"adaptive", "constant-18"}:
        return BaselineProvider(identifier)
    if kind == "openrouter":
        return _HTTPProvider(identifier, api_key_env=api_key_env or "OPENROUTER_API_KEY", base_url=base_url or "https://openrouter.ai/api/v1")
    if kind == "openai":
        return _HTTPProvider(identifier, api_key_env=api_key_env or "OPENAI_API_KEY", base_url=base_url or "https://api.openai.com/v1")
    if kind == "compat":
        if not base_url:
            raise ProviderError("compat provider requires --base-url")
        return _HTTPProvider(identifier, api_key_env=api_key_env or "SCB_API_KEY", base_url=base_url)
    if kind == "hf":
        return HFProvider(identifier, adapter=adapter)
    raise ProviderError(f"unknown model provider {kind!r}")
