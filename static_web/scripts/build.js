import {
  cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const STATIC = resolve(ROOT, "static_web");
const DIST = resolve(ROOT, "dist");
const PAGES = resolve(DIST, "cloudflare-pages");
const SPACE = resolve(DIST, "huggingface-space");

function loadJson(relative) {
  return JSON.parse(readFileSync(resolve(ROOT, relative), "utf8"));
}

function traceCatalog() {
  const trainedPath = "artifacts/live_y_qwen35_4b_rl/eval_held_out.json";
  if (existsSync(resolve(ROOT, trainedPath))) {
    const artifact = loadJson(trainedPath);
    const seeds = artifact.episodes.map((episode) => ({
      id: episode.id,
      split: episode.split,
      seed_set: episode.seed_set,
      seed_index: episode.seed_index,
      master_seed_hex: episode.master_seed_hex,
      episode_id: episode.episode_id,
      actions: episode.actions,
      local_total_cost: episode.local_total_cost,
      paired_base_stock_local_total_cost: episode.paired_base_stock_local_total_cost,
      reward: episode.reward,
    })).sort((left, right) => left.seed_index - right.seed_index);
    if (seeds.length !== 10) throw new Error(`expected 10 trained-Qwen traces, found ${seeds.length}`);
    return {
      schema_version: "1.1.0",
      environment_version: "0.2.0",
      scenario_id: "t5-strategic-y-v2",
      controlled_role: "wholesaler",
      model: artifact.model,
      evaluation_summary: artifact.summary,
      seeds,
    };
  }
  const sources = [
    ["development", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_development/results.json"],
    ["validation", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_validation_controls/results.json"],
  ];
  const seeds = sources.flatMap(([split, path]) => loadJson(path).episodes
    .filter((episode) => episode.scenario_id === "t5-strategic-y-v2")
    .map((episode) => ({
      id: `${split}-${episode.seed_index}`,
      split,
      seed_index: episode.seed_index,
      master_seed_hex: episode.master_seed_hex,
      episode_id: episode.episode_id,
      actions: episode.actions,
      local_total_cost: episode.local_total_cost,
      paired_base_stock_local_total_cost: episode.paired_base_stock_local_total_cost,
      reward: episode.reward,
    })));
  seeds.sort((left, right) => ({ development: 0, validation: 1 })[left.split] - ({ development: 0, validation: 1 })[right.split] || left.seed_index - right.seed_index);
  if (seeds.length !== 8) throw new Error(`expected 8 recorded fallback traces, found ${seeds.length}`);
  return {
    schema_version: "1.0.0",
    environment_version: "0.2.0",
    scenario_id: "t5-strategic-y-v2",
    controlled_role: "wholesaler",
    model: "Recorded LLM",
    seeds,
  };
}

function writeRuntimeConfig(target) {
  const loggingEndpoint = process.env.PUBLIC_LOGGING_ENDPOINT || "";
  writeFileSync(
    resolve(target, "config.js"),
    `globalThis.BEER_GAME_CONFIG=${JSON.stringify({ loggingEndpoint })};\n`,
  );
}

function buildTarget(target) {
  mkdirSync(target, { recursive: true });
  cpSync(resolve(STATIC, "public"), target, { recursive: true });
  cpSync(resolve(STATIC, "src/app.js"), resolve(target, "app.js"));
  cpSync(resolve(STATIC, "src/telemetry.js"), resolve(target, "telemetry.js"));
  cpSync(resolve(STATIC, "src/styles.css"), resolve(target, "styles.css"));
  cpSync(resolve(STATIC, "src/sim"), resolve(target, "sim"), { recursive: true });
  mkdirSync(resolve(target, "data"), { recursive: true });
  writeFileSync(
    resolve(target, "data/llm-comparison.json"),
    `${JSON.stringify(traceCatalog(), null, 2)}\n`,
  );
  writeRuntimeConfig(target);
}

if (existsSync(DIST)) rmSync(DIST, { recursive: true });
buildTarget(PAGES);
cpSync(PAGES, SPACE, { recursive: true });
writeFileSync(
  resolve(SPACE, "README.md"),
  `---\ntitle: Beer Distribution Game\nemoji: 🍺\ncolorFrom: yellow\ncolorTo: red\nsdk: static\napp_file: index.html\npinned: false\n---\n\n# Beer Distribution Game\n\nPublic static human baseline for the Y-network beer game.\n\n- **Seat:** wholesaler only\n- **Horizon:** 36 weeks + settlement\n- **Orders:** integers 0–128\n- **Factory capacity:** 400 (feasible upstream supply)\n- **Environment:** deterministic / replayable JS port of v0.2.0 mechanics\n\nPlay locally after \`npm run build\`, or on Cloudflare Pages. The Prime Intellect / Verifiers Hub package still publishes frozen Tier-5 capacity **22** for historical leaderboard parity — do not mix those scores with this Space.\n`,
);
console.log(`Built ${PAGES}`);
console.log(`Built ${SPACE}`);
