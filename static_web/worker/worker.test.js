import { beforeEach, describe, expect, test } from "vitest";
import { env } from "cloudflare:workers";
import { applyD1Migrations, SELF } from "cloudflare:test";
import { BeerEpisode, TRAINING_PROTOCOL_ID, trainingScenario } from "../src/sim/index.js";

const ORIGIN = "https://game.example";

function makeRecord(actions, status = "completed") {
  const spec = trainingScenario("0123456789abcdef", 0);
  const episode = new BeerEpisode(spec, "wholesaler", {
    includeReference: status === "completed",
  });
  episode.start();
  const weekly = [];
  for (const action of actions) {
    episode.placeOrder(action);
    const transition = episode.operationalTransitions.at(-1);
    const state = transition.states_after_fulfillment.wholesaler;
    weekly.push({
      week: transition.week,
      inventory: state.inventory,
      backlog: state.backlog,
      local_cost: transition.local_costs.wholesaler,
    });
  }
  const grade = status === "completed" ? episode.outcome.grade : null;
  return {
    session_uuid: "018f9bc3-0feb-4ca2-b395-a7c1e0779877",
    timestamp: "2026-07-25T20:00:00.000Z",
    env_version: TRAINING_PROTOCOL_ID,
    tier: 5,
    role: "wholesaler",
    split: "training",
    seed_index: 0,
    seed: spec.master_seed_hex,
    scenario_id: spec.scenario_id,
    variant: "headline",
    prior_beer_game_experience: "unsure",
    status,
    completed: status === "completed",
    actions,
    weekly,
    final_total_cost: grade?.primary.local_total_cost ?? null,
    base_stock_cost: grade?.primary.paired_base_stock_local_total_cost ?? null,
    episode_reward: grade?.episode_reward ?? null,
  };
}

async function post(record, origin = ORIGIN) {
  return SELF.fetch("https://logger.example/session", {
    method: "POST",
    headers: { Origin: origin, "Content-Type": "text/plain;charset=UTF-8" },
    body: JSON.stringify(record),
  });
}

beforeEach(async () => {
  await applyD1Migrations(env.DB, env.TEST_MIGRATIONS);
});

describe("anonymous logging Worker", () => {
  test("replays and stores a canonical completed record in local D1", async () => {
    const record = makeRecord(Array(36).fill(8));
    const response = await post(record);
    expect(response.status).toBe(202);

    const row = await env.DB.prepare(
      "SELECT * FROM human_sessions WHERE session_uuid = ?",
    ).bind(record.session_uuid).first();
    expect(row).toMatchObject({
      session_uuid: record.session_uuid,
      env_version: TRAINING_PROTOCOL_ID,
      tier: 5,
      role: "wholesaler",
      status: "completed",
      completed: 1,
    });
    expect(row.final_total_cost).toBe(record.final_total_cost);
    expect(row.base_stock_cost).toBe(record.base_stock_cost);
    expect(JSON.parse(row.actions_json)).toEqual(record.actions);
    expect(JSON.parse(row.weekly_json)).toEqual(record.weekly);
  });

  test("completion supersedes abandonment and cannot be overwritten later", async () => {
    const abandoned = makeRecord([0, 128], "abandoned");
    expect((await post(abandoned)).status).toBe(202);
    const completed = makeRecord(Array(36).fill(8));
    expect((await post(completed)).status).toBe(202);
    expect((await post(abandoned)).status).toBe(202);
    const row = await env.DB.prepare(
      "SELECT status, completed, actions_json FROM human_sessions WHERE session_uuid = ?",
    ).bind(completed.session_uuid).first();
    expect(row.status).toBe("completed");
    expect(row.completed).toBe(1);
    expect(JSON.parse(row.actions_json)).toHaveLength(36);
  });

  test("rejects schema, privacy, replay, origin, and read-endpoint violations", async () => {
    const base = makeRecord(Array(36).fill(8));
    expect((await post({ ...base, email: "not-allowed@example.com" })).status).toBe(400);
    expect((await post({ ...base, weekly: base.weekly.map((row, index) => (
      index === 0 ? { ...row, backlog: row.backlog + 1 } : row
    )) })).status).toBe(400);
    expect((await post(base, "https://evil.example")).status).toBe(403);
    expect((await SELF.fetch("https://logger.example/session", {
      headers: { Origin: ORIGIN },
    })).status).toBe(405);
    expect((await SELF.fetch("https://logger.example/records", {
      headers: { Origin: ORIGIN },
    })).status).toBe(404);
  });
});
