from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from beer_distribution_rl.agents.llm.prefix_kv_cache import (
    PrefixKVCache,
    UnsupportedCacheLayout,
)


class FakeTokenizer:
    rows = {
        "a": [1, 2, 3, 4, 5],
        "b": [1, 2, 3, 4, 6],
    }

    def __call__(self, text, **_kwargs):
        return {"input_ids": self.rows[text]}

    def decode(self, ids, **_kwargs):
        return "decoded:" + ",".join(map(str, ids))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.prefix_forwards = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.prefix_forwards += 1
        value = torch.zeros((input_ids.shape[0], 1, input_ids.shape[1]))
        return SimpleNamespace(past_key_values=((value, value.clone()),))

    def generate(self, input_ids, **_kwargs):
        extra = torch.full((input_ids.shape[0], 2), 9, dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


def test_reuses_shared_prefix_and_returns_original_prompt_ids():
    model = FakeModel()
    cache = PrefixKVCache(model, min_prefix_tokens=3)
    tokenizer = FakeTokenizer()

    first = cache.generate(
        tokenizer,
        ["a", "b"],
        prompt_max_tokens=32,
        max_new_tokens=4,
        sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=0,
        eos_token_id=9,
    )
    second = cache.generate(
        tokenizer,
        ["a", "b"],
        prompt_max_tokens=32,
        max_new_tokens=4,
        sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=0,
        eos_token_id=9,
    )

    assert first is not None and second is not None
    assert [row[0] for row in first] == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]
    assert all(row[1] == [9] for row in second)
    assert model.prefix_forwards == 1
    stats = cache.snapshot()
    assert stats["cache_hits"] == 1
    assert stats["prefix_tokens_avoided"] > 0


def test_short_shared_prefix_falls_back():
    cache = PrefixKVCache(FakeModel(), min_prefix_tokens=10)
    result = cache.generate(
        FakeTokenizer(),
        ["a", "b"],
        prompt_max_tokens=32,
        max_new_tokens=4,
        sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=0,
        eos_token_id=9,
    )
    assert result is None
    assert cache.snapshot()["fallbacks"] == 1


class DynamicishLayer:
    """Stands in for transformers' ``DynamicLayer``: growing key/value tensors."""

    def __init__(self, batch: int = 1):
        self.keys = torch.zeros((batch, 2, 3, 4))
        self.values = torch.zeros((batch, 2, 3, 4))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)


class LinearAttentionishLayer:
    """Stands in for transformers' ``LinearAttentionLayer``.

    The real class carries ``conv_states`` / ``recurrent_states`` guarded by
    ``is_*_initialized`` flags and, crucially, has **no**
    ``batch_repeat_interleave`` -- which is what broke the A40 smoke.
    """

    def __init__(self, batch: int = 1, recurrent: bool = True, keyed: bool = False):
        conv = torch.zeros((batch, 5, 6))
        rec = torch.zeros((batch, 7, 8)) if recurrent else None
        if keyed:
            # transformers >= 5.14 keys each state by index inside one layer.
            self.conv_states = {0: conv}
            self.is_conv_states_initialized = {0: True}
            self.recurrent_states = {0: rec}
            self.is_recurrent_states_initialized = {0: recurrent}
        else:
            # transformers 5.9 stores a single tensor per layer.
            self.conv_states = conv
            self.is_conv_states_initialized = True
            self.recurrent_states = rec
            self.is_recurrent_states_initialized = recurrent


class OpaqueLayer:
    """A layer this implementation genuinely cannot expand."""


class HybridCache:
    """Mimics ``Cache``: delegates to layers, and blows up on the first that cannot."""

    def __init__(self, layers):
        self.layers = list(layers)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)


class HybridModel(FakeModel):
    def __init__(self, layers_factory):
        super().__init__()
        self.layers_factory = layers_factory

    def forward(self, input_ids, attention_mask, use_cache):
        self.prefix_forwards += 1
        return SimpleNamespace(past_key_values=HybridCache(self.layers_factory()))


def _generate(cache, prompts=("a", "b")):
    return cache.generate(
        FakeTokenizer(),
        list(prompts),
        prompt_max_tokens=32,
        max_new_tokens=4,
        sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=0,
        eos_token_id=9,
    )


def test_hybrid_cache_expands_layers_that_lack_batch_repeat_interleave():
    """Regression: Qwen3.5 mixes full-attention and linear-attention layers.

    ``Cache.batch_repeat_interleave`` delegates per layer, and
    ``LinearAttentionLayer`` does not implement it, so the delegating call
    raised ``AttributeError`` partway through and forced a fallback on every
    multi-row batch.
    """
    layers = [DynamicishLayer(), LinearAttentionishLayer()]
    expanded = PrefixKVCache._repeat_cache(HybridCache(layers), 3)

    dynamic, linear = expanded.layers
    assert dynamic.keys.shape[0] == 3
    assert dynamic.values.shape[0] == 3
    assert linear.conv_states.shape[0] == 3
    assert linear.recurrent_states.shape[0] == 3
    # The originals must be untouched: the entry stays cached for reuse.
    assert layers[0].keys.shape[0] == 1
    assert layers[1].conv_states.shape[0] == 1


def test_hybrid_model_generation_does_not_fall_back():
    model = HybridModel(lambda: [DynamicishLayer(), LinearAttentionishLayer()])
    cache = PrefixKVCache(model, min_prefix_tokens=3)

    result = _generate(cache)

    assert result is not None
    stats = cache.snapshot()
    assert stats["fallbacks"] == 0
    assert stats["enabled"] is True
    assert stats["last_error"] is None


def test_uninitialized_recurrent_state_is_skipped_not_repeated():
    layer = LinearAttentionishLayer(recurrent=False)
    expanded = PrefixKVCache._repeat_cache(HybridCache([layer]), 2)
    assert expanded.layers[0].conv_states.shape[0] == 2
    assert expanded.layers[0].recurrent_states is None


def test_single_row_batch_needs_no_expansion():
    layer = LinearAttentionishLayer()
    expanded = PrefixKVCache._repeat_cache(HybridCache([layer]), 1)
    assert expanded.layers[0].conv_states.shape[0] == 1


def test_unsupported_layout_disables_the_cache_instead_of_retrying():
    """An unexpandable layout is permanent, so stop paying a prefix forward for it."""
    model = HybridModel(lambda: [OpaqueLayer()])
    cache = PrefixKVCache(model, min_prefix_tokens=3)

    assert _generate(cache) is None
    forwards_after_first = model.prefix_forwards
    assert cache.disabled
    assert "OpaqueLayer" in cache.snapshot()["disabled_reason"]

    assert _generate(cache) is None
    # The second attempt must not have run the prefix through the model again.
    assert model.prefix_forwards == forwards_after_first
    assert cache.snapshot()["enabled"] is False


def test_transient_failure_does_not_disable_the_cache():
    """Only a layout problem is permanent; a one-off error must stay recoverable."""

    class FlakyModel(FakeModel):
        def forward(self, input_ids, attention_mask, use_cache):
            self.prefix_forwards += 1
            raise RuntimeError("CUDA hiccup")

    cache = PrefixKVCache(FlakyModel(), min_prefix_tokens=3)
    assert _generate(cache) is None
    assert not cache.disabled
    assert cache.snapshot()["enabled"] is True


def test_unsupported_cache_layout_is_raised_for_unknown_layer_types():
    with pytest.raises(UnsupportedCacheLayout):
        PrefixKVCache._repeat_layer(OpaqueLayer(), 2)


def test_real_transformers_hybrid_cache_is_expanded():
    """Pin the fix against the actual classes, not just stand-ins.

    Reproduces the A40 smoke failure verbatim -- transformers'
    ``Cache.batch_repeat_interleave`` raises
    ``AttributeError: 'LinearAttentionLayer' object has no attribute
    'batch_repeat_interleave'`` on a Qwen3.5-style hybrid layer stack -- and
    asserts our per-layer path handles it.
    """
    import copy

    cache_utils = pytest.importorskip("transformers.cache_utils")
    for name in ("DynamicLayer", "LinearAttentionLayer", "DynamicCache"):
        if not hasattr(cache_utils, name):
            pytest.skip(f"transformers build has no {name}")

    dynamic = cache_utils.DynamicLayer()
    dynamic.lazy_initialization(torch.zeros((1, 2, 3, 4)), torch.zeros((1, 2, 3, 4)))
    dynamic.keys = torch.zeros((1, 2, 3, 4))
    dynamic.values = torch.zeros((1, 2, 3, 4))

    linear = cache_utils.LinearAttentionLayer()
    linear.conv_states = torch.zeros((1, 5, 6))
    linear.is_conv_states_initialized = True
    linear.recurrent_states = torch.zeros((1, 7, 8))
    linear.is_recurrent_states_initialized = True

    cache = cache_utils.DynamicCache()
    cache.layers = [dynamic, linear]

    with pytest.raises(AttributeError, match="batch_repeat_interleave"):
        copy.deepcopy(cache).batch_repeat_interleave(3)

    expanded = PrefixKVCache._repeat_cache(cache, 3)
    grown_dynamic, grown_linear = expanded.layers
    assert grown_dynamic.keys.shape[0] == 3
    assert grown_dynamic.values.shape[0] == 3
    assert grown_linear.conv_states.shape[0] == 3
    assert grown_linear.recurrent_states.shape[0] == 3
    assert dynamic.keys.shape[0] == 1
    assert linear.conv_states.shape[0] == 1


def test_layer_carrying_both_state_families_expands_both():
    """Guards a silent-corruption path, not just a slow one.

    ``LinearAttentionAndFullAttentionLayer`` inherits ``batch_repeat_interleave``
    from ``DynamicLayer``, which repeats only keys and values.  Delegating to it
    would leave the recurrent state at the original batch size, so generation
    would run on a mismatched cache instead of raising.
    """

    class BothFamiliesLayer(LinearAttentionishLayer):
        def __init__(self):
            super().__init__()
            self.keys = torch.zeros((1, 2, 3, 4))
            self.values = torch.zeros((1, 2, 3, 4))

        def batch_repeat_interleave(self, repeats: int) -> None:
            # Deliberately incomplete, exactly like the inherited implementation.
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    expanded = PrefixKVCache._repeat_cache(HybridCache([BothFamiliesLayer()]), 4)
    layer = expanded.layers[0]
    assert layer.keys.shape[0] == 4
    assert layer.values.shape[0] == 4
    assert layer.conv_states.shape[0] == 4
    assert layer.recurrent_states.shape[0] == 4


def test_real_hybrid_layer_class_expands_both_state_families():
    cache_utils = pytest.importorskip("transformers.cache_utils")
    if not hasattr(cache_utils, "LinearAttentionAndFullAttentionLayer"):
        pytest.skip("transformers build has no LinearAttentionAndFullAttentionLayer")

    layer = cache_utils.LinearAttentionAndFullAttentionLayer()
    layer.keys = torch.zeros((1, 2, 3, 4))
    layer.values = torch.zeros((1, 2, 3, 4))
    layer.conv_states = torch.zeros((1, 5, 6))
    layer.is_conv_states_initialized = True
    layer.recurrent_states = torch.zeros((1, 7, 8))
    layer.is_recurrent_states_initialized = True

    cache = cache_utils.DynamicCache()
    cache.layers = [layer]
    grown = PrefixKVCache._repeat_cache(cache, 2).layers[0]

    assert grown.keys.shape[0] == 2
    assert grown.conv_states.shape[0] == 2
    assert grown.recurrent_states.shape[0] == 2


def test_keyed_state_dicts_are_expanded():
    """transformers 5.14 stores conv/recurrent states as dicts keyed by index.

    Regression: the first version of this fix was written against 5.9, where
    they are plain tensors.  On 5.14 it called ``repeat_interleave`` on a dict
    and every multi-row batch silently fell back to uncached generation.
    """
    layer = LinearAttentionishLayer(keyed=True)
    expanded = PrefixKVCache._repeat_cache(HybridCache([layer]), 3).layers[0]

    assert isinstance(expanded.conv_states, dict)
    assert expanded.conv_states[0].shape[0] == 3
    assert expanded.recurrent_states[0].shape[0] == 3
    assert layer.conv_states[0].shape[0] == 1


def test_keyed_state_dict_skips_uninitialized_entries():
    layer = LinearAttentionishLayer(keyed=True, recurrent=False)
    expanded = PrefixKVCache._repeat_cache(HybridCache([layer]), 2).layers[0]
    assert expanded.conv_states[0].shape[0] == 2
    assert expanded.recurrent_states[0] is None


def test_real_linear_attention_layer_expands_on_this_transformers_version():
    """Version-agnostic: populate the real class however this build shapes it."""
    cache_utils = pytest.importorskip("transformers.cache_utils")
    if not hasattr(cache_utils, "LinearAttentionLayer"):
        pytest.skip("transformers build has no LinearAttentionLayer")

    layer = cache_utils.LinearAttentionLayer()
    keyed = isinstance(layer.conv_states, dict)
    conv = torch.zeros((1, 5, 6))
    rec = torch.zeros((1, 7, 8))
    if keyed:
        keys = list(layer.conv_states) or [0]
        layer.conv_states = dict.fromkeys(keys, conv)
        layer.recurrent_states = dict.fromkeys(keys, rec)
        layer.is_conv_states_initialized = dict.fromkeys(keys, True)
        layer.is_recurrent_states_initialized = dict.fromkeys(keys, True)
    else:
        layer.conv_states, layer.is_conv_states_initialized = conv, True
        layer.recurrent_states, layer.is_recurrent_states_initialized = rec, True

    cache = cache_utils.DynamicCache()
    cache.layers = [layer]
    grown = PrefixKVCache._repeat_cache(cache, 4).layers[0]

    got_conv = grown.conv_states[next(iter(grown.conv_states))] if keyed else grown.conv_states
    got_rec = (
        grown.recurrent_states[next(iter(grown.recurrent_states))] if keyed else grown.recurrent_states
    )
    assert got_conv.shape[0] == 4
    assert got_rec.shape[0] == 4
