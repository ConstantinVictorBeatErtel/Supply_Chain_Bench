from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.live_y_integrity as integrity
import scripts.train_colab_grpo_wholesaler as pilot
import scripts.train_qwen35_live_y_rl as live_y
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_for


def _live_args(**overrides):
    values = {
        "mode": "train",
        "adapter": None,
        "output_dir": "unused",
        "updates": 0,
        "group_size": 1,
        "train_minibatch": 1,
        "learning_rate": 5e-6,
        "seed": 1,
        "allow_held_out_test": False,
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_frozen_live_y_splits_are_derived_unique_and_disjoint():
    splits = integrity.load_seed_splits()

    assert {name: len(values) for name, values in splits.items()} == {
        "train": 16,
        "research_development": 10,
        "held_out_test": 20,
    }
    assert not set(splits["train"]) & set(splits["research_development"])
    assert not set(splits["train"]) & set(splits["held_out_test"])
    assert not set(splits["research_development"]) & set(splits["held_out_test"])


def test_new_test_split_does_not_overlap_any_existing_benchmark_seed_manifest():
    splits = integrity.load_seed_splits()
    candidate_hex = set(splits["held_out_test"])
    candidate_int = {int(seed, 16) for seed in candidate_hex}

    frontier = json.loads(
        (integrity.ROOT / "eval" / "frontier_t5_y_36w_100_seed_split.json").read_text()
    )
    serial = json.loads((integrity.ROOT / "eval" / "held_out_seeds.json").read_text())
    robustness = json.loads(
        (integrity.ROOT / "eval" / "robustness_seeds.json").read_text()
    )
    normative = {
        scenario_for(5, split, index, "headline").master_seed_hex
        for split, count in (("development", 3), ("validation", 5), ("test", 10))
        for index in range(count)
    }

    assert candidate_hex.isdisjoint(frontier["seeds"])
    assert candidate_hex.isdisjoint(normative)
    assert candidate_int.isdisjoint(map(int, serial["seeds"]))
    assert candidate_int.isdisjoint(map(int, robustness["seeds"]))


def test_held_out_test_is_opt_in_and_forbidden_for_model_selection():
    splits = integrity.load_seed_splits()
    test_seed = splits["held_out_test"][0]

    with pytest.raises(PermissionError, match="explicit"):
        integrity.seeds_for("held_out_test")
    assert integrity.seeds_for("held_out_test", allow_held_out_test=True) == splits["held_out_test"]
    for purpose in (
        "training_rollout",
        "training_data_generation",
        "calibration",
        "development_evaluation",
        "prompt_tuning",
        "hyperparameter_selection",
        "early_stopping",
    ):
        with pytest.raises(AssertionError, match="forbidden"):
            integrity.assert_seed_use([test_seed], purpose)


def test_train_and_development_seeds_cannot_cross_purposes():
    splits = integrity.load_seed_splits()
    integrity.assert_seed_use(splits["train"], "training_rollout")
    integrity.assert_seed_use(splits["research_development"], "development_evaluation")

    with pytest.raises(AssertionError, match="forbidden"):
        integrity.assert_seed_use([splits["train"][0]], "development_evaluation")
    with pytest.raises(AssertionError, match="forbidden"):
        integrity.assert_seed_use([splits["research_development"][0]], "training_data_generation")


def test_sft_rows_reject_development_and_test_seed_leakage():
    splits = integrity.load_seed_splits()
    train_rows = [{"meta": {"seed": splits["train"][0]}}]
    integrity.assert_training_data_rows(train_rows, require_live_y_split=True)

    for forbidden in (
        splits["research_development"][0],
        splits["held_out_test"][0],
    ):
        with pytest.raises(AssertionError, match="forbidden"):
            integrity.assert_training_data_rows([{"meta": {"seed": forbidden}}])
    with pytest.raises(ValueError, match="meta.seed"):
        integrity.assert_training_data_rows([{"meta": {}}], require_live_y_split=True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"quantity":0}', 0),
        (' {"quantity": 128}\n', 128),
        ("12", None),
        ('{"quantity":12.0}', None),
        ('{"quantity":true}', None),
        ('{"quantity":12,"reason":"x"}', None),
        ("place_order(quantity=12)", None),
        ('{"quantity":129}', None),
        ('prefix {"quantity":12}', None),
    ],
)
def test_policy_parser_is_one_strict_interface(text, expected):
    assert integrity.parse_policy_action(text) == expected


def test_policy_contract_is_native_y_wholesaler_with_settlement():
    protocol = integrity.load_policy_protocol()
    assert "20-week serial" in protocol["separate_from"]
    spec = scenario_for(5, "development", 0, "headline")
    integrity.assert_policy_scenario(spec)

    episodes = [BeerEpisode(spec, "wholesaler") for _ in range(2)]
    for episode in episodes:
        observation = episode.start()
        while not episode.done:
            result = episode.place_order(8)
            if not result["done"]:
                observation = result["next_observation"]
        assert observation is not None
        assert len(episode.operational_transitions) == 36
        assert len(episode.settlement_transitions) == 3
    assert episodes[0].outcome["counterparties"] == episodes[1].outcome["counterparties"]
    assert episodes[0].outcome["grade"] == episodes[1].outcome["grade"]


def test_live_train_mode_routes_pre_post_evaluation_to_ten_development_seeds(monkeypatch):
    args = _live_args()
    live_y.configure_pilot(args)
    pilot_args = pilot.parse_args()

    train_tasks = pilot.load_tasks("development", pilot_args.train_seeds)
    development_tasks = pilot.development_tasks_for_training(pilot_args)
    assert len(train_tasks) == 16
    assert len(development_tasks) == 10
    assert {task.data.scenario["master_seed_hex"] for task in train_tasks} == set(
        integrity.load_seed_splits()["train"]
    )
    assert {task.data.scenario["master_seed_hex"] for task in development_tasks} == set(
        integrity.load_seed_splits()["research_development"]
    )


def test_pilot_main_uses_development_hook_for_both_pre_and_post(monkeypatch, tmp_path):
    args = SimpleNamespace(
        tier5_controls=False,
        eval_only=False,
        adapter=None,
        seed=1,
        output_dir=str(tmp_path),
        model_name="fake",
        train_seeds=list(range(16)),
        eval_split="research_development",
        eval_seeds=list(range(10)),
        group_size=1,
        updates=0,
        dry_run=False,
        learning_rate=1e-5,
    )
    train_tasks = [f"train-{i}" for i in range(16)]
    development_tasks = [f"development-{i}" for i in range(10)]
    evaluated = []

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        def parameters(self):
            return [SimpleNamespace(requires_grad=True)]

        def save_pretrained(self, _path):
            pass

    class FakeTokenizer:
        def save_pretrained(self, _path):
            pass

    monkeypatch.setattr(pilot, "parse_args", lambda: args)
    monkeypatch.setattr(pilot, "set_seed", lambda _seed: None)
    monkeypatch.setattr(pilot, "load_tasks", lambda *_a, **_kw: train_tasks)
    monkeypatch.setattr(pilot, "development_tasks_for_training", lambda _args: development_tasks)
    monkeypatch.setattr(pilot, "load_policy", lambda _args: (FakeModel(), FakeTokenizer()))
    monkeypatch.setattr(
        pilot,
        "evaluate",
        lambda _model, _tokenizer, tasks, _args: evaluated.append(list(tasks)) or {"summary": {}},
    )
    monkeypatch.setattr(pilot, "save_json", lambda _path, _value: None)
    monkeypatch.setattr(
        pilot,
        "torch",
        SimpleNamespace(optim=SimpleNamespace(AdamW=lambda *_a, **_kw: object())),
    )

    pilot.main()

    assert evaluated == [development_tasks, development_tasks]
    assert all(set(tasks).isdisjoint(train_tasks) for tasks in evaluated)


def test_live_held_out_mode_requires_explicit_opt_in():
    with pytest.raises(PermissionError, match="explicit"):
        live_y.configure_pilot(_live_args(mode="held-out-test"))

    live_y.configure_pilot(
        _live_args(mode="held-out-test", allow_held_out_test=True)
    )
    args = pilot.parse_args()
    tasks = pilot.load_tasks(args.eval_split, args.eval_seeds)
    assert len(tasks) == 20
    assert {task.data.scenario["master_seed_hex"] for task in tasks} == set(
        integrity.load_seed_splits()["held_out_test"]
    )


def test_split_manifest_is_not_the_serial_benchmark_manifest():
    live = json.loads(integrity.SPLIT_PATH.read_text())
    serial = json.loads((integrity.ROOT / "eval" / "held_out_seeds.json").read_text())
    assert live["split_id"] == integrity.SPLIT_ID
    assert serial["schema_version"] == "1.0"
    assert live["environment"].startswith("v0.2.0 Tier-5 Y")
    assert len(serial["seeds"]) == 100
