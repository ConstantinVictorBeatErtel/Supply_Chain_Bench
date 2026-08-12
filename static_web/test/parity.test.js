import { describe, expect, test } from "vitest";
import {
  BeerEpisode, PythonRandom, episodeId, masterSeedHex, replayActions,
  scenarioFor, stableStringify,
} from "../src/sim/index.js";
import { artifactRows, pythonOracle, traceArtifact } from "./helpers.js";

function jsResult(caseRow) {
  const base = scenarioFor(5, caseRow.split, caseRow.seed_index);
  const spec = caseRow.capacity === undefined ? base : { ...base, capacity: caseRow.capacity };
  const { episode, observations } = replayActions(
    spec, "wholesaler", caseRow.actions,
  );
  return {
    episode_id: episode.episodeId,
    observations,
    operational_transitions: episode.operationalTransitions,
    settlement_transitions: episode.settlementTransitions,
    terminal_inventory_positions: episode.outcome.terminal_inventory_positions,
    final_state: episode.outcome.final_state,
    grade: episode.outcome.grade,
  };
}

describe("Python/JavaScript parity", () => {
  test("seed, RNG, and episode identity vectors match CPython", () => {
    expect(masterSeedHex("development", 0)).toBe("e2056d6d52741a08");
    const rng = new PythonRandom(123n);
    expect(rng.random()).toBe(0.052363598850944326);
    expect(rng.random()).toBe(0.08718667752263232);
    expect(rng.gauss()).toBe(-0.3985847224936054);
    expect(rng.gauss()).toBe(0.26274798902660246);
    const spec = scenarioFor(5, "development", 0);
    expect(episodeId(spec, "wholesaler")).toBe(
      "sha256:f789d5921b14e4f692db05571b9650eba10805ee9a7751650dd8a43a1c3c3d72",
    );
  });

  test("all playable seeds match the frozen Python environment", () => {
    const traces = artifactRows();
    const { capacity } = traceArtifact();
    const cases = traces.flatMap((row, rowIndex) => [
      { split: row.split, seed_index: row.seed_index, actions: row.actions },
      // The debrief replays the recorded model at the public capacity.
      { split: row.split, seed_index: row.seed_index, actions: row.actions, capacity },
      {
        split: row.split,
        seed_index: row.seed_index,
        actions: Array.from({ length: 36 }, (_, week) => week % 2 ? 128 : 0),
      },
      {
        split: row.split,
        seed_index: row.seed_index,
        actions: Array.from({ length: 36 }, (_, week) => (week * 17 + rowIndex * 11) % 129),
      },
    ]);
    const expected = pythonOracle(cases);
    const actual = cases.map(jsResult);
    expect(actual).toEqual(expected);
  });
});

describe("boundary behavior and determinism", () => {
  test("0 and 128 are accepted and invalid orders do not mutate state or RNG", () => {
    const episode = new BeerEpisode(scenarioFor(5, "development", 0), "wholesaler");
    episode.start();
    const beforeState = episode.core.canonicalSnapshot();
    const beforePrepared = stableStringify(episode.core.prepared);
    const beforeRng = stableStringify(episode.core.demand.rng.snapshot());
    for (const invalid of [-1, 129, 1.5, NaN, Infinity, "8", null, true]) {
      expect(() => episode.placeOrder(invalid)).toThrow();
      expect(episode.core.canonicalSnapshot()).toBe(beforeState);
      expect(stableStringify(episode.core.prepared)).toBe(beforePrepared);
      expect(stableStringify(episode.core.demand.rng.snapshot())).toBe(beforeRng);
    }
    episode.placeOrder(0);
    episode.placeOrder(128);
    expect(episode.operationalTransitions.map((row) => row.orders.wholesaler)).toEqual([0, 128]);
  });

  test("a fixed trajectory is byte-identical across 100 runs", () => {
    const actions = Array.from({ length: 36 }, (_, week) => (week * 23 + 7) % 129);
    const outputs = Array.from({ length: 100 }, () => {
      const { episode } = replayActions(
        scenarioFor(5, "validation", 4), "wholesaler", actions,
      );
      return stableStringify(episode.outcome);
    });
    expect(new Set(outputs).size).toBe(1);
  });
});
