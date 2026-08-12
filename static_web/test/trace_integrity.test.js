import { describe, expect, test } from "vitest";
import { replayActions, scenarioFor } from "../src/sim/index.js";
import { artifactRows, traceArtifact } from "./helpers.js";

describe("LLM trace integrity", () => {
  test("the recorded comparison traces replay to their published totals", () => {
    const { capacity } = traceArtifact();
    const rows = artifactRows();
    expect(rows).toHaveLength(8);
    for (const row of rows) {
      const spec = { ...scenarioFor(5, row.split, row.seed_index), capacity };
      const { episode } = replayActions(spec, "wholesaler", row.actions);
      const grade = episode.outcome.grade;
      expect(episode.episodeId).toBe(row.episode_id);
      expect(episode.spec.master_seed_hex).toBe(row.master_seed_hex);
      expect(grade.primary.local_total_cost).toBe(row.local_total_cost);
      expect(grade.primary.paired_base_stock_local_total_cost)
        .toBe(row.paired_base_stock_local_total_cost);
      expect(grade.episode_reward).toBe(row.reward);
      expect(grade.costs.system_total_cost).toBe(row.system_total_cost);
    }
  });

  test("every recorded week carries the order it was written with", () => {
    for (const row of artifactRows()) {
      expect(row.weeks).toHaveLength(row.actions.length);
      row.weeks.forEach((entry, index) => {
        expect(entry.week).toBe(index + 1);
        expect(entry.quantity).toBe(row.actions[index]);
        expect(typeof entry.thought).toBe("string");
      });
    }
  });
});
