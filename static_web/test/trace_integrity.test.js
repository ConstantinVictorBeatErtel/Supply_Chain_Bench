import { describe, expect, test } from "vitest";
import { replayActions, scenarioFor } from "../src/sim/index.js";
import { artifactRows } from "./helpers.js";

describe("LLM trace integrity", () => {
  test("the recorded comparison traces replay to their published totals", () => {
    const rows = artifactRows();
    expect(rows).toHaveLength(8);
    for (const row of rows) {
      const { episode } = replayActions(
        scenarioFor(5, row.split, row.seed_index), "wholesaler", row.actions,
      );
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
});
