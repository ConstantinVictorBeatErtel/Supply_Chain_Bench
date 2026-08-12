"""Token-level surrogate for the wholesaler GRPO trainer.

The archived run scored the mean log probability over the whole completion, so
the seven boilerplate tokens of ``{"quantity": 48}`` outvoted the one token
carrying the order.  These tests pin the replacement: a per-token ratio, a
double-sided clip, and a loss restricted to the digits of the order.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_colab_grpo_wholesaler import (  # noqa: E402
    ActionRecord,
    decision_token_mask,
    loss_token_mask,
    train_update,
)

VOCAB = 32


class FakeTokenizer:
    """Ids 0-9 decode to their own digit; everything else is punctuation."""

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(str(i) if 0 <= i <= 9 else "{" for i in ids)


class ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB, 8)
        self.head = torch.nn.Linear(8, VOCAB)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def _record(completion: list[int], advantage: float, group: int = 0) -> ActionRecord:
    return ActionRecord(
        prompt_ids=[10, 11, 12],
        completion_ids=completion,
        group_id=group,
        raw_text="",
        quantity=1,
        valid=True,
        advantage=advantage,
        action_token_mask=decision_token_mask(FakeTokenizer(), completion),
    )


def _args(**overrides):
    base = dict(
        train_minibatch=2,
        inference_minibatch=2,
        clip_epsilon=0.2,
        dual_clip=3.0,
        all_completion_tokens=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_decision_token_mask_selects_only_digit_tokens():
    # {  "  quantity  "  :  4  8  }
    completion = [20, 21, 22, 21, 23, 4, 8, 24]
    assert decision_token_mask(FakeTokenizer(), completion) == [
        False, False, False, False, False, True, True, False
    ]


def test_loss_token_mask_falls_back_when_no_digit_token_exists():
    record = _record([20, 21, 24], advantage=1.0)
    assert not any(record.action_token_mask)
    # A completion with no order in it is a protocol failure; it still has to
    # carry its penalty, so the whole span is scored rather than nothing.
    assert loss_token_mask(record, decision_tokens_only=True) == [True, True, True]


def test_loss_token_mask_can_be_widened_to_the_whole_completion():
    record = _record([20, 4, 8, 24], advantage=1.0)
    assert loss_token_mask(record, decision_tokens_only=True) == [False, True, True, False]
    assert loss_token_mask(record, decision_tokens_only=False) == [True] * 4


def test_train_update_scores_only_the_order_digits():
    torch.manual_seed(0)
    model = ToyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    records = [_record([20, 21, 4, 8, 24], 1.0), _record([20, 21, 3, 2, 24], -1.0)]

    stats = train_update(model, optimizer, records, _args())

    # Two digits per completion, two completions.
    assert stats["scored_tokens"] == 4.0
    assert stats["trainable_actions"] == 2.0
    assert stats["loss"] == pytest.approx(stats["loss"])  # finite, not NaN


def test_train_update_scores_every_token_when_widened():
    torch.manual_seed(0)
    model = ToyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    records = [_record([20, 21, 4, 8, 24], 1.0), _record([20, 21, 3, 2, 24], -1.0)]

    stats = train_update(model, optimizer, records, _args(all_completion_tokens=True))

    assert stats["scored_tokens"] == 10.0


def test_train_update_moves_parameters_and_reports_clip_fraction():
    torch.manual_seed(0)
    model = ToyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.head.weight.detach().clone()
    records = [_record([20, 4, 24], 1.0), _record([20, 8, 24], -1.0)]

    stats = train_update(model, optimizer, records, _args())

    assert not torch.allclose(before, model.head.weight)
    assert 0.0 <= stats["clip_fraction"] <= 1.0
    # The first minibatch is exactly on-policy, so no token can be clipped yet.
    assert stats["clip_fraction"] == 0.0


def test_zero_advantage_records_are_skipped():
    model = ToyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = train_update(model, optimizer, [_record([20, 4, 24], 0.0)], _args())
    assert stats["trainable_actions"] == 0.0
    assert stats["scored_tokens"] == 0.0


def test_dual_clip_bounds_the_negative_advantage_surrogate():
    """A far off-policy token with a negative advantage cannot dominate."""
    advantage = torch.tensor([[-2.0]])
    ratio = torch.tensor([[50.0]])  # wildly off-policy
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    bounded = torch.where(advantage < 0, torch.maximum(surrogate, 3.0 * advantage), surrogate)
    assert surrogate.item() == pytest.approx(-100.0)
    assert bounded.item() == pytest.approx(-6.0)
