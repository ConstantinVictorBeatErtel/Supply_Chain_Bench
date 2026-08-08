from types import SimpleNamespace

import torch

from beer_distribution_rl.agents.llm.prefix_kv_cache import PrefixKVCache


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
