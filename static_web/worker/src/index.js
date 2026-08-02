import {
  BeerEpisode, PLAYABLE_SEEDS, scenarioFor, stableStringify,
} from "../../src/sim/index.js";

const MAX_BODY_BYTES = 64 * 1024;
const ALLOWED_FIELDS = new Set([
  "session_uuid", "timestamp", "env_version", "tier", "role", "split",
  "seed_index", "seed", "scenario_id", "variant",
  "prior_beer_game_experience", "status", "completed", "actions", "weekly",
  "final_total_cost", "base_stock_cost", "episode_reward",
]);
const WEEKLY_FIELDS = new Set(["week", "inventory", "backlog", "local_cost"]);
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const UTC_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

function response(status, body, origin = "") {
  const headers = {
    "Content-Type": "application/json;charset=UTF-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  };
  if (origin) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers.Vary = "Origin";
  }
  return new Response(JSON.stringify(body), { status, headers });
}

function allowedOrigins(env) {
  return new Set(String(env.ALLOWED_ORIGINS || "")
    .split(",").map((value) => value.trim()).filter(Boolean));
}

function originFor(request, env) {
  const origin = request.headers.get("Origin") || "";
  return allowedOrigins(env).has(origin) ? origin : "";
}

function exactKeys(value, allowed) {
  return value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).every((key) => allowed.has(key));
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isNullableNumber(value) {
  return value === null || isFiniteNumber(value);
}

function isPlayableSeed(split, seedIndex) {
  return PLAYABLE_SEEDS.some((seed) => seed.split === split && seed.seedIndex === seedIndex);
}

function validateShape(record) {
  if (!exactKeys(record, ALLOWED_FIELDS)
      || Object.keys(record).length !== ALLOWED_FIELDS.size) {
    return "payload fields are invalid";
  }
  if (!UUID_V4.test(record.session_uuid)
      || typeof record.timestamp !== "string"
      || !UTC_TIMESTAMP.test(record.timestamp)
      || !Number.isFinite(Date.parse(record.timestamp))
      || new Date(record.timestamp).toISOString() !== record.timestamp
      || record.env_version !== "0.2.0"
      || record.tier !== 5
      || record.role !== "wholesaler"
      || record.variant !== "headline"
      || !isPlayableSeed(record.split, record.seed_index)
      || !["yes", "no", "unsure"].includes(record.prior_beer_game_experience)
      || !["completed", "abandoned"].includes(record.status)
      || typeof record.completed !== "boolean"
      || record.completed !== (record.status === "completed")
      || !Array.isArray(record.actions)
      || record.actions.length > 36
      || record.actions.some((value) => !Number.isInteger(value) || value < 0 || value > 128)
      || !Array.isArray(record.weekly)
      || record.weekly.length !== record.actions.length
      || record.weekly.some((row) => !exactKeys(row, WEEKLY_FIELDS)
        || Object.keys(row).length !== WEEKLY_FIELDS.size
        || !Number.isInteger(row.week)
        || !Number.isInteger(row.inventory)
        || !Number.isInteger(row.backlog)
        || !isFiniteNumber(row.local_cost))
      || !isNullableNumber(record.final_total_cost)
      || !isNullableNumber(record.base_stock_cost)
      || !isNullableNumber(record.episode_reward)) {
    return "payload values are invalid";
  }
  if (record.completed && record.actions.length !== 36) return "completed games require 36 actions";
  if (!record.completed && [record.final_total_cost, record.base_stock_cost, record.episode_reward]
    .some((value) => value !== null)) {
    return "abandoned games cannot include scores";
  }
  return "";
}

function replay(record) {
  const spec = scenarioFor(5, record.split, record.seed_index);
  if (record.seed !== spec.master_seed_hex
      || record.scenario_id !== spec.scenario_id) {
    throw new Error("scenario identifiers do not match");
  }
  const episode = new BeerEpisode(spec, "wholesaler", {
    includeReference: record.completed,
  });
  episode.start();
  const weekly = [];
  for (const action of record.actions) {
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
  if (stableStringify(weekly) !== stableStringify(record.weekly)) {
    throw new Error("weekly trajectory does not match replay");
  }
  if (!record.completed) {
    return { episode, weekly, finalTotalCost: null, baseStockCost: null, reward: null };
  }
  const grade = episode.outcome.grade;
  const finalTotalCost = grade.primary.local_total_cost;
  const baseStockCost = grade.primary.paired_base_stock_local_total_cost;
  const reward = grade.episode_reward;
  if (record.final_total_cost !== finalTotalCost
      || record.base_stock_cost !== baseStockCost
      || record.episode_reward !== reward) {
    throw new Error("submitted scores do not match replay");
  }
  return { episode, weekly, finalTotalCost, baseStockCost, reward };
}

async function writeRecord(env, record, verified) {
  const statement = `
    INSERT INTO human_sessions (
      session_uuid, timestamp, env_version, tier, role, split, seed_index, seed,
      scenario_id, variant, prior_beer_game_experience, actions_json, weekly_json,
      final_total_cost, base_stock_cost, episode_reward, status, completed
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_uuid) DO UPDATE SET
      timestamp = excluded.timestamp,
      env_version = excluded.env_version,
      tier = excluded.tier,
      role = excluded.role,
      split = excluded.split,
      seed_index = excluded.seed_index,
      seed = excluded.seed,
      scenario_id = excluded.scenario_id,
      variant = excluded.variant,
      prior_beer_game_experience = excluded.prior_beer_game_experience,
      actions_json = excluded.actions_json,
      weekly_json = excluded.weekly_json,
      final_total_cost = excluded.final_total_cost,
      base_stock_cost = excluded.base_stock_cost,
      episode_reward = excluded.episode_reward,
      status = excluded.status,
      completed = excluded.completed
    WHERE human_sessions.completed = 0 AND excluded.completed = 1
  `;
  await env.DB.prepare(statement).bind(
    record.session_uuid,
    new Date(record.timestamp).toISOString(),
    "0.2.0",
    5,
    "wholesaler",
    record.split,
    record.seed_index,
    verified.episode.spec.master_seed_hex,
    verified.episode.spec.scenario_id,
    "headline",
    record.prior_beer_game_experience,
    stableStringify(record.actions),
    stableStringify(verified.weekly),
    verified.finalTotalCost,
    verified.baseStockCost,
    verified.reward,
    record.status,
    record.completed ? 1 : 0,
  ).run();
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/session") return response(404, { error: "not found" });
    const origin = originFor(request, env);
    if (!origin) return response(403, { error: "origin not allowed" });
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": origin,
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
          Vary: "Origin",
        },
      });
    }
    if (request.method !== "POST") return response(405, { error: "method not allowed" }, origin);

    const declaredLength = Number(request.headers.get("Content-Length") || 0);
    if (declaredLength > MAX_BODY_BYTES) {
      return response(413, { error: "payload too large" }, origin);
    }
    const body = await request.text();
    if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) {
      return response(413, { error: "payload too large" }, origin);
    }
    let record;
    try {
      record = JSON.parse(body);
    } catch {
      return response(400, { error: "invalid JSON" }, origin);
    }
    const shapeError = validateShape(record);
    if (shapeError) return response(400, { error: shapeError }, origin);
    try {
      const verified = replay(record);
      await writeRecord(env, record, verified);
      return response(202, { accepted: true }, origin);
    } catch {
      return response(400, { error: "record does not match deterministic replay" }, origin);
    }
  },
};
