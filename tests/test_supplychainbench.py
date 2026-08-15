from __future__ import annotations

import json
from pathlib import Path

import pytest

from supplychainbench.eval import evaluate
from supplychainbench.leaderboard import build_leaderboard, load_entries
from supplychainbench.providers import ProviderError, create_provider
from supplychainbench.results import ResultValidationError, aggregate_episode_rows, validate_result, write_atomic
from supplychainbench.suites import continual_episode_jobs, episode_jobs, expected_seeds, prompt_for, project_observation, reference_cost


def test_standard_manifest_is_frozen_and_disjoint_from_new_suites():
    standard = set(expected_seeds("standard"))
    assert len(standard) == 16
    for suite in ("demand_shift", "unknown_lead_time", "capacity_shock", "supply_disruption", "held_out_dynamics"):
        seeds = set(expected_seeds(suite))
        assert len(seeds) == 16
        assert seeds.isdisjoint(standard)


def test_standard_reference_and_constant_baseline_are_reproducible(tmp_path: Path):
    first = evaluate("agent:constant-18", "standard", output=tmp_path / "first.json", resume=False)
    second = evaluate("agent:constant-18", "standard", output=tmp_path / "second.json", resume=False)
    assert first["aggregate"]["normalized_score"] == pytest.approx(19.819722683458046)
    assert first["aggregate"] == second["aggregate"]
    assert [row["actions"] for row in first["episodes"]] == [row["actions"] for row in second["episodes"]]


def test_hidden_suite_schedule_is_deterministic_and_not_prompt_visible():
    job = episode_jobs("capacity_shock")[0]
    system, user = prompt_for(job, project_observation(job, {
        "episode_id": "private", "scenario_id": "private", "constraints": {"factory_capacity": 24},
        "state": {"inventory_on_hand": 1},
    }))
    assert system == prompt_for(job, project_observation(job, {"constraints": {}, "state": {}}))[0]
    text = system + user
    for secret in ("factory_capacity", "capacity_after", "change_week", str(job.ground_truth.get("capacity_after"))):
        assert secret not in text
    assert episode_jobs("capacity_shock")[0].ground_truth == episode_jobs("capacity_shock")[0].ground_truth


def test_dynamic_capacity_schedule_changes_production_only_after_hidden_event():
    job = episode_jobs("capacity_shock")[0]
    assert job.runtime.capacity_for_week(job.event_week - 1, job.spec.capacity) == 400
    assert job.runtime.capacity_for_week(job.event_week, job.spec.capacity) in {24, 32}
    assert reference_cost(job)["local_total_cost"] > 0


@pytest.mark.parametrize("suite", ["demand_shift", "capacity_shock", "supply_disruption"])
def test_event_suites_have_seeded_timing_and_deterministic_aware_reference(suite: str):
    first, second = episode_jobs(suite)[0], episode_jobs(suite)[0]
    assert first.event_week == second.event_week
    assert reference_cost(first) == reference_cost(second)
    assert 13 <= int(first.event_week) <= 23
    if suite == "supply_disruption":
        assert first.ground_truth["disruption_duration"] in {3, 4, 5}
        runtime = first.runtime
        start = first.ground_truth["disruption_start"]
        duration = first.ground_truth["disruption_duration"]
        assert runtime.capacity_for_week(start, 400) == 0
        assert runtime.capacity_for_week(start + duration, 400) == 400


def test_unknown_lead_time_uses_variable_settlement_and_continual_world_is_fixed():
    job = episode_jobs("unknown_lead_time")[0]
    assert (job.spec.order_delay, job.spec.shipment_delay) != (1, 2)
    assert job.spec.settlement_weeks == job.spec.order_delay + job.spec.shipment_delay
    jobs = continual_episode_jobs("held_out_dynamics", "adaptation", 3)
    assert len({json.dumps(job.ground_truth, sort_keys=True) for job in jobs}) == 1
    assert len({job.seed for job in jobs}) == 3


def test_result_schema_rejects_missing_failure_and_accepts_partial_diagnostic():
    rows = [{"seed": "a", "protocol_clean": False, "failure": "invalid", "local_total_cost": None, "reference_cost": 1.0}]
    aggregate = aggregate_episode_rows(rows)
    payload = {"schema_version": "1.0.0", "benchmark": {}, "model": {}, "suite": {"id": "standard"},
               "episodes": rows, "aggregate": aggregate, "run": {}, "configuration": {}}
    validate_result(payload, expected_suite="standard")
    bad = dict(payload)
    bad["episodes"] = [{"seed": "a", "protocol_clean": False, "local_total_cost": None}]
    with pytest.raises(ResultValidationError):
        validate_result(bad, expected_suite="standard")


def _result(seed: str, *, clean: bool = True, status: str = "complete", superseded: bool = False) -> dict:
    row = {"seed": seed, "protocol_clean": clean, "failure": None if clean else "invalid",
           "local_total_cost": 10.0 if clean else None, "reference_cost": 1.0}
    rows = [row]
    aggregate = aggregate_episode_rows(rows)
    return {"schema_version": "1.0.0", "benchmark": {}, "model": {"identifier": seed, "label": seed},
            "suite": {"id": "standard", "version": "1.0.0", "expected_seeds": [seed]},
            "episodes": rows, "aggregate": aggregate, "run": {"status": status}, "configuration": {},
            "provenance": {"publication_status": "superseded" if superseded else "active"}}


def test_leaderboard_rejects_malformed_input_before_replacing_outputs(tmp_path: Path):
    (tmp_path / "standard").mkdir()
    (tmp_path / "standard" / "bad.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(ResultValidationError):
        build_leaderboard(tmp_path, tracked_only=False)


def test_leaderboard_unranks_failed_and_superseded_runs_and_order_is_deterministic(tmp_path: Path):
    target = tmp_path / "standard"
    target.mkdir()
    seeds = expected_seeds("standard")
    for name, payload in {
        "z.json": _result(seeds[0]),
        "a.json": _result(seeds[0]),
        "failed.json": _result(seeds[1], clean=False),
        "old.json": _result(seeds[2], superseded=True),
    }.items():
        # A leaderboard entry must cover the full frozen manifest; duplicate the
        # same valid row shape across all seeds for this isolated ordering test.
        payload["suite"]["expected_seeds"] = list(seeds)
        payload["model"] = {"identifier": name[:-5], "label": name[:-5]}
        payload["episodes"] = [{**payload["episodes"][0], "seed": seed} for seed in seeds]
        payload["aggregate"] = aggregate_episode_rows(payload["episodes"])
        write_atomic(target / name, payload)
    entries, invalid = load_entries(tmp_path, tracked_only=False)
    assert not invalid
    board = build_leaderboard(tmp_path, tracked_only=False)
    assert [item["model"] for item in board["suites"]["standard"]["ranked"]] == ["a", "z"]
    assert {item["unranked_reason"] for item in board["suites"]["standard"]["unranked"]} == {"protocol failures", "superseded"}
    assert len(entries) == 4


def test_atomic_writer_replaces_file_without_leftover_temp(tmp_path: Path):
    path = tmp_path / "nested" / "result.json"
    write_atomic(path, {"value": 1})
    write_atomic(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 2}
    assert list(path.parent.glob(".*.tmp")) == []


def test_api_provider_is_lazy_and_reports_missing_credentials(monkeypatch):
    monkeypatch.delenv("SCB_TEST_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="SCB_TEST_API_KEY"):
        create_provider("compat:model", base_url="https://example.invalid", api_key_env="SCB_TEST_API_KEY")
