"""Safe reusable-prefix KV caching for batched decoder-only generation.

The weekly observation changes, so a whole-turn KV cache cannot be reused.
This module caches only the longest token prefix shared by the current batch
(normally the system prompt and chat-template prefix).  A defensive fallback
is intentional: model/cache API differences must never change experiment
correctness or stop a run.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


class UnsupportedCacheLayout(TypeError):
    """The model's KV cache cannot be batch-expanded by this implementation.

    Distinguished from a transient failure (an allocator hiccup, a model-version
    quirk) because it is a permanent property of the loaded model: retrying it
    on every batch costs a prefix forward pass each time and can never succeed.
    """


class PrefixKVCache:
    """Cache model KV states for invariant prompt prefixes.

    Entries are valid only while the model weights are unchanged.  Call
    :meth:`clear` after every optimizer update.  The cache is deliberately
    small because a training run normally has one invariant system prefix.
    """

    def __init__(self, model: Any, *, min_prefix_tokens: int = 64, max_entries: int = 8) -> None:
        if min_prefix_tokens < 1:
            raise ValueError("min_prefix_tokens must be positive")
        self.model = model
        self.min_prefix_tokens = min_prefix_tokens
        self.max_entries = max_entries
        self._entries: dict[tuple[int, ...], Any] = {}
        self._stats = defaultdict(int)
        self._last_error: str | None = None
        self._disabled_reason: str | None = None

    def clear(self) -> None:
        self._entries.clear()
        self._stats["clears"] += 1
        self._last_error = None

    @property
    def disabled(self) -> bool:
        return self._disabled_reason is not None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": not self.disabled,
            "entries": len(self._entries),
            "cache_lookups": self._stats["lookups"],
            "cache_hits": self._stats["hits"],
            "prefix_forward_calls": self._stats["prefix_forward_calls"],
            "fallbacks": self._stats["fallbacks"],
            "prefix_tokens_avoided": self._stats["prefix_tokens_avoided"],
            "last_error": self._last_error,
            "disabled_reason": self._disabled_reason,
        }

    @staticmethod
    def _common_prefix(token_rows: list[list[int]]) -> int:
        if not token_rows:
            return 0
        length = min(len(row) for row in token_rows)
        for index in range(length):
            token = token_rows[0][index]
            if any(row[index] != token for row in token_rows[1:]):
                return index
        return length

    @staticmethod
    def _clone_cache(cache: Any) -> Any:
        """Clone a cache before generation, which appends new KVs in-place."""

        if isinstance(cache, tuple):
            return tuple(
                tuple(value.clone() if hasattr(value, "clone") else copy.deepcopy(value) for value in layer)
                if isinstance(layer, tuple)
                else copy.deepcopy(layer)
                for layer in cache
            )
        return copy.deepcopy(cache)

    @staticmethod
    def _repeat_tensor(value: Any, repeats: int) -> Any:
        if value is None or not hasattr(value, "repeat_interleave"):
            return value
        return value.repeat_interleave(repeats, dim=0)

    @classmethod
    def _repeat_state(cls, state: Any, initialized: Any, repeats: int) -> Any:
        """Expand one recurrent-state container, whatever shape the version uses."""

        if isinstance(state, dict):
            expanded = dict(state)
            for key, value in state.items():
                ready = initialized.get(key, True) if isinstance(initialized, dict) else initialized
                if ready is False:
                    continue
                expanded[key] = cls._repeat_tensor(value, repeats)
            return expanded
        if initialized is False:
            return state
        return cls._repeat_tensor(state, repeats)

    @classmethod
    def _repeat_layer(cls, layer: Any, repeats: int) -> None:
        """Expand one cache layer along the batch dimension, in place.

        Transformers' ``Cache.batch_repeat_interleave`` delegates to each layer,
        and not every layer type implements it.  Qwen3.5 is hybrid: its
        full-attention layers are ``DynamicLayer`` (which does) while its linear
        attention layers are ``LinearAttentionLayer`` (which does not), so the
        delegating call raises partway through the layer loop.  Handling the
        layers individually keeps the cache usable on hybrid models.
        """

        handled = False

        # Linear-attention layers keep a fixed-size recurrent state instead of a
        # growing key/value pair.  ``reorder_cache`` on the same classes treats
        # those tensors as batch-first and guards on the same initialization
        # flags, so this mirrors the layer's own notion of its batch dimension.
        #
        # The container shape is version-dependent: transformers 5.9 stores one
        # tensor per layer, 5.14 stores ``dict[int, Tensor | None]`` keyed by
        # state index (with the initialization flags keyed the same way).  Both
        # are handled rather than pinning a version.
        for name in ("conv_states", "recurrent_states"):
            if not hasattr(layer, name):
                continue
            handled = True
            setattr(
                layer,
                name,
                cls._repeat_state(
                    getattr(layer, name),
                    getattr(layer, f"is_{name}_initialized", None),
                    repeats,
                ),
            )

        # Both families are expanded rather than one or the other.  A layer can
        # carry both -- transformers' ``LinearAttentionAndFullAttentionLayer``
        # does -- and it *inherits* ``batch_repeat_interleave`` from
        # ``DynamicLayer``, which touches only keys and values.  Delegating to
        # that method would leave the recurrent state at the original batch size
        # and corrupt generation silently rather than raising.
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            handled = True
            if layer.keys is not None:
                layer.keys = layer.keys.repeat_interleave(repeats, dim=0)
            if layer.values is not None:
                layer.values = layer.values.repeat_interleave(repeats, dim=0)

        if handled:
            return

        # Only for layouts we do not recognize at all: trust the layer to know
        # how to expand itself.
        repeat = getattr(layer, "batch_repeat_interleave", None)
        if callable(repeat):
            repeat(repeats)
            return

        raise UnsupportedCacheLayout(
            f"cache layer {type(layer).__name__!r} cannot be batch-expanded"
        )

    @classmethod
    def _repeat_cache(cls, cache: Any, batch_size: int) -> Any:
        cache = cls._clone_cache(cache)
        if batch_size == 1:
            return cache
        # Prefer per-layer expansion over the delegating top-level helper: the
        # latter mutates layers in order and leaves the cache half-expanded when
        # one layer type is unsupported.
        layers = getattr(cache, "layers", None)
        if layers is not None:
            for layer in layers:
                cls._repeat_layer(layer, batch_size)
            return cache
        repeat = getattr(cache, "batch_repeat_interleave", None)
        if callable(repeat):
            result = repeat(batch_size)
            return cache if result is None else result
        if isinstance(cache, tuple):
            return tuple(
                tuple(value.repeat_interleave(batch_size, dim=0) for value in layer)
                if isinstance(layer, tuple)
                else layer
                for layer in cache
            )
        raise UnsupportedCacheLayout(f"unsupported past-key-value cache type: {type(cache)!r}")

    def _prefix_cache(self, prefix_ids: list[int], device: Any) -> Any:
        key = tuple(prefix_ids)
        self._stats["lookups"] += 1
        if key in self._entries:
            self._stats["hits"] += 1
            return self._entries[key]
        import torch

        ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            outputs = self.model(input_ids=ids, attention_mask=mask, use_cache=True)
        past = getattr(outputs, "past_key_values", None)
        if past is None and isinstance(outputs, dict):
            past = outputs.get("past_key_values")
        if past is None:
            raise RuntimeError("model did not return past_key_values for prefix caching")
        if len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = past
        self._stats["prefix_forward_calls"] += 1
        return past

    def generate(
        self,
        tokenizer: Any,
        prompts: list[str],
        *,
        prompt_max_tokens: int,
        max_new_tokens: int,
        sample: bool,
        temperature: float,
        top_p: float,
        pad_token_id: int | None,
        eos_token_id: int | list[int] | None,
    ) -> list[tuple[list[int], list[int], str]] | None:
        """Generate from a cached prefix, or return ``None`` for safe fallback."""

        if not prompts:
            return []
        if self.disabled:
            self._stats["fallbacks"] += 1
            return None
        try:
            rows: list[list[int]] = []
            for prompt in prompts:
                encoded = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=prompt_max_tokens,
                    add_special_tokens=False,
                )
                ids = encoded["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                rows.append([int(token) for token in ids])

            common = self._common_prefix(rows)
            prefix_len = common
            # Generation needs at least one uncached prompt token.  Otherwise
            # the returned sequence has no unambiguous prompt/completion split.
            if prefix_len == min(len(row) for row in rows):
                prefix_len -= 1
            if prefix_len < self.min_prefix_tokens:
                self._stats["fallbacks"] += 1
                return None

            prefix_ids = rows[0][:prefix_len]
            cache_was_hit = tuple(prefix_ids) in self._entries
            device = next(self.model.parameters()).device
            cached = self._prefix_cache(prefix_ids, device)
            outputs: list[tuple[list[int], list[int], str] | None] = [None] * len(rows)

            # Equal suffix lengths avoid right-padding after the cached prefix,
            # which would otherwise shift decoder positions for short rows.
            groups: dict[int, list[int]] = defaultdict(list)
            for index, row in enumerate(rows):
                groups[len(row) - prefix_len].append(index)
            for suffix_len, indices in groups.items():
                import torch

                suffix = torch.tensor(
                    [rows[index][prefix_len:] for index in indices],
                    dtype=torch.long,
                    device=device,
                )
                attention = torch.ones(
                    (len(indices), prefix_len + suffix_len), dtype=torch.long, device=device
                )
                past = self._repeat_cache(cached, len(indices))
                self.model.eval()
                with torch.inference_mode():
                    generated = self.model.generate(
                        input_ids=suffix,
                        attention_mask=attention,
                        past_key_values=past,
                        do_sample=sample,
                        temperature=temperature if sample else 1.0,
                        top_p=top_p if sample else 1.0,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=True,
                    )
                for row_number, original_index in enumerate(indices):
                    completion = generated[row_number, suffix_len:].detach().cpu().tolist()
                    if eos_token_id is not None:
                        eos_values = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
                        for eos in eos_values:
                            if eos in completion:
                                completion = completion[: completion.index(eos) + 1]
                                break
                    while completion and completion[-1] == pad_token_id:
                        completion.pop()
                    outputs[original_index] = (
                        rows[original_index],
                        completion,
                        tokenizer.decode(completion, skip_special_tokens=True),
                    )

            self._stats["prefix_tokens_avoided"] += prefix_len * (
                len(rows) if cache_was_hit else max(0, len(rows) - 1)
            )
            return [output for output in outputs if output is not None]
        except UnsupportedCacheLayout as exc:
            # Permanent for this model, so stop paying a prefix forward pass per
            # batch to rediscover it.  Correctness is unaffected: the caller
            # falls back to plain batched generation exactly as before.
            self._stats["fallbacks"] += 1
            self._last_error = f"{type(exc).__name__}: {exc}"[:240]
            self._disabled_reason = self._last_error
            self._entries.clear()
            return None
        except Exception as exc:  # pragma: no cover - exercised by model-version fallback
            self._stats["fallbacks"] += 1
            self._last_error = f"{type(exc).__name__}: {exc}"[:240]
            return None
