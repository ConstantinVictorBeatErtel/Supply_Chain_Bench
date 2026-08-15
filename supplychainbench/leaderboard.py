"""Validate result files and render deterministic SupplyChainBench leaderboards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from supplychainbench.results import ResultValidationError, validate_result, write_atomic
from supplychainbench.suites import expected_seeds

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"


def _tracked(path: Path) -> bool:
    try:
        result = subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except (OSError, ValueError):
        return False


def _result_paths(root: Path, tracked_only: bool) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*/**/*.json")):
        if path.name in {"leaderboard.json", "manifest.json"}:
            continue
        if tracked_only and not _tracked(path):
            continue
        paths.append(path)
    return paths


def _rank_state(payload: dict[str, Any]) -> tuple[bool, str | None]:
    suite = payload["suite"]["id"]
    expected = set(expected_seeds(suite))
    actual = {str(row["seed"]) for row in payload["episodes"]}
    if payload.get("provenance", {}).get("publication_status") == "superseded":
        return False, "superseded"
    if payload["run"].get("status") != "complete" or actual != expected:
        return False, "incomplete seed coverage"
    if payload["aggregate"].get("protocol_failure_episodes", 0):
        return False, "protocol failures"
    return True, None


def load_entries(results_root: Path = RESULTS_ROOT, *, tracked_only: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    entries: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in _result_paths(results_root, tracked_only):
        try:
            payload = json.loads(path.read_text())
            validate_result(payload, expected_suite=payload.get("suite", {}).get("id"))
        except (OSError, json.JSONDecodeError, ResultValidationError, TypeError, KeyError) as exc:
            invalid.append({"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "error": str(exc)})
            continue
        ranked, reason = _rank_state(payload)
        entry = {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                 "model": payload["model"].get("label", payload["model"]["identifier"]),
                 "model_identifier": payload["model"]["identifier"], "suite": payload["suite"]["id"],
                 "ranked": ranked, "unranked_reason": reason,
                 "aggregate": payload["aggregate"], "run": payload["run"],
                 "publication_status": payload.get("provenance", {}).get("publication_status", "active")}
        entries.append(entry)
    return entries, invalid


def build_leaderboard(results_root: Path = RESULTS_ROOT, *, tracked_only: bool = True) -> dict[str, Any]:
    entries, invalid = load_entries(results_root, tracked_only=tracked_only)
    if invalid:
        details = "; ".join(f"{item['path']}: {item['error']}" for item in invalid)
        raise ResultValidationError(f"invalid result files prevent leaderboard generation: {details}")
    suites: dict[str, dict[str, Any]] = {}
    for suite in sorted({entry["suite"] for entry in entries}):
        subset = [entry for entry in entries if entry["suite"] == suite]
        ranked = [entry for entry in subset if entry["ranked"] and entry["aggregate"].get("normalized_score") is not None]
        ranked.sort(key=lambda entry: (-float(entry["aggregate"]["normalized_score"]), float(entry["aggregate"].get("mean_local_cost") or float("inf")), entry["model_identifier"].lower()))
        unranked = [entry for entry in subset if not entry["ranked"]]
        unranked.sort(key=lambda entry: (entry["unranked_reason"] or "", entry["model_identifier"].lower()))
        suites[suite] = {"ranked": ranked, "unranked": unranked}
    return {"schema_version": "1.0.0", "benchmark": {"id": "supplychainbench", "version": "1.0.0"}, "suites": suites, "invalid": []}


def _markdown(board: dict[str, Any]) -> str:
    lines = ["# SupplyChainBench leaderboard", "", "Only complete, fully protocol-clean active runs receive ranks. Unranked runs remain visible with their failure/coverage reason.", ""]
    for suite, data in board["suites"].items():
        lines += [f"## {suite}", "", "| Rank | Model | Normalized score | Mean cost | Clean | Status |", "| ---: | --- | ---: | ---: | ---: | --- |"]
        for rank, entry in enumerate(data["ranked"], 1):
            agg = entry["aggregate"]
            lines.append(f"| {rank} | `{entry['model']}` | {agg['normalized_score']:.2f} | {agg['mean_local_cost']:.2f} | {agg['protocol_clean_episodes']}/{agg['episodes_attempted']} | ranked |")
        for entry in data["unranked"]:
            agg = entry["aggregate"]
            score = "—" if agg.get("normalized_score") is None else f"{agg['normalized_score']:.2f}"
            lines.append(f"| — | `{entry['model']}` | {score} | {agg.get('mean_local_cost') if agg.get('mean_local_cost') is not None else '—'} | {agg['protocol_clean_episodes']}/{agg['episodes_attempted']} | {entry['unranked_reason']} |")
        if not data["ranked"] and not data["unranked"]:
            lines.append("| — | (no valid results) | — | — | — | — |")
        lines.append("")
    return "\n".join(lines)


def _chart(board: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("leaderboard chart requires matplotlib; install `pip install -e '.[benchmark]'`") from exc
    rows = [(suite, entry["model"], entry["aggregate"]["normalized_score"])
            for suite, data in board["suites"].items() for entry in data["ranked"]]
    if not rows:
        rows = [(suite, "(no ranked runs)", 0.0) for suite in board["suites"]]
    rows.sort(key=lambda row: (row[0], -float(row[2]), row[1].lower()))
    labels = [f"{suite}: {model}" for suite, model, _ in rows]
    values = [float(value) for _, _, value in rows]
    fig, ax = plt.subplots(figsize=(11, max(3, 0.38 * len(rows) + 1.4)))
    ax.barh(labels[::-1], values[::-1], color="#0b5d6b")
    ax.set_xlabel("Normalized score (suite-local; higher is better)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=output.suffix,
                                         dir=output.parent, delete=False) as temp:
            temp_name = temp.name
        fig.savefig(temp_name, dpi=160)
        os.replace(temp_name, output)
        temp_name = None
    finally:
        plt.close(fig)
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, mode="w", encoding="utf-8",
                                         delete=False) as temp:
            temp_name = temp.name
            temp.write(text)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated SupplyChainBench leaderboards")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--include-untracked", action="store_true",
                        help="also scan untracked result files (default scans tracked files only)")
    parser.add_argument("--tracked-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, default=RESULTS_ROOT / "leaderboard.json")
    parser.add_argument("--markdown", type=Path, default=RESULTS_ROOT / "leaderboard.md")
    parser.add_argument("--chart", type=Path, default=ROOT / "docs/assets/supplychainbench-leaderboard.png")
    args = parser.parse_args()
    board = build_leaderboard(args.results_root, tracked_only=not args.include_untracked or args.tracked_only)
    write_atomic(args.json, board)
    _write_text_atomic(args.markdown, _markdown(board))
    _chart(board, args.chart)
    for suite, data in board["suites"].items():
        for rank, entry in enumerate(data["ranked"], 1):
            print(f"{suite}\t{rank}\t{entry['model']}\t{entry['aggregate']['normalized_score']:.2f}")
    print(f"wrote {args.json}")
    print(f"wrote {args.markdown}")
    print(f"wrote {args.chart}")


if __name__ == "__main__":
    main()
