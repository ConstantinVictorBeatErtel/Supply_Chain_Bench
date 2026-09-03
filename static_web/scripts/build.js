import {
  cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { replayActions, standardResearchScenario } from "../src/sim/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const STATIC = resolve(ROOT, "static_web");
const DIST = resolve(ROOT, "dist");
const PAGES = resolve(DIST, "cloudflare-pages");
const SPACE = resolve(DIST, "huggingface-space");

function loadJson(relative) {
  return JSON.parse(readFileSync(resolve(ROOT, relative), "utf8"));
}

/** Frozen 16-seed capacity-400 evaluation of the trained Qwen policy. */
export const QWEN_TRACE_PATH = "results/standard/qwen3.5-4b-grpo.json";
export const BENCHMARK_REPLAY_PATH = "static_web/public/data/benchmark-replay.json";

function traceCatalog() {
  const artifact = loadJson(QWEN_TRACE_PATH);
  const bucketIndexes = new Map();
  const seeds = artifact.episodes.map((episode) => {
    const seedIndex = bucketIndexes.get(episode.bucket) || 0;
    bucketIndexes.set(episode.bucket, seedIndex + 1);
    if (!episode.protocol_clean || episode.actions.length !== 36) {
      throw new Error(`trained-Qwen trace ${episode.seed} is not a clean 36-week episode`);
    }
    const scenario = standardResearchScenario(episode.bucket, episode.seed, seedIndex);
    const replay = replayActions(scenario, "wholesaler", episode.actions).episode;
    const localTotal = replay.outcome.grade.primary.local_total_cost;
    if (localTotal !== episode.local_total_cost) {
      throw new Error(`trained-Qwen replay ${episode.seed} drifted: ${localTotal} != ${episode.local_total_cost}`);
    }
    let runningCost = 0;
    const costsOverTime = replay.histories.wholesaler.map((week) => {
      runningCost += Number(week.local_cost);
      return runningCost;
    });
    return {
      id: `${episode.bucket}-${seedIndex}`,
      split: scenario.split,
      seed_set: "supplychainbench-standard-v1",
      seed_index: seedIndex,
      master_seed_hex: episode.seed,
      episode_id: replay.episodeId,
      scenario_id: scenario.scenario_id,
      bucket: episode.bucket,
      scenario,
      actions: episode.actions,
      costs_over_time: costsOverTime,
      local_total_cost: localTotal,
      paired_base_stock_local_total_cost: replay.outcome.grade.primary.paired_base_stock_local_total_cost,
      reward: replay.outcome.grade.episode_reward,
      benchmark_reference_cost: episode.reference_cost,
      benchmark_score: episode.score,
      system_total_cost: replay.outcome.grade.costs.system_total_cost,
    };
  });
  if (seeds.length !== 16) throw new Error(`expected 16 trained-Qwen traces, found ${seeds.length}`);
  return {
    schema_version: "2.0.0",
    environment_version: "live-y-domain-randomized-grpo-v1",
    scenario_id: "supplychainbench-standard-v1",
    controlled_role: "wholesaler",
    capacity: 400,
    model: artifact.model.identifier,
    model_label: artifact.model.label,
    adapter: artifact.configuration.adapter,
    evaluation_summary: artifact.aggregate,
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
  if (existsSync(resolve(ROOT, BENCHMARK_REPLAY_PATH))) {
    cpSync(resolve(ROOT, BENCHMARK_REPLAY_PATH), resolve(target, "data/benchmark-replay.json"));
  }
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
  `---\ntitle: Beer Distribution Game\nemoji: 🍺\ncolorFrom: yellow\ncolorTo: red\nsdk: static\napp_file: index.html\npinned: false\n---\n\n# Beer Distribution Game\n\nPublic human baseline for the stochastic Y-network training environment.\n\n- **Seat:** wholesaler only\n- **Horizon:** 36 weeks + settlement\n- **Orders:** integers 0–128\n- **Factory capacity:** 400 (feasible upstream supply)\n- **Environment:** live-y-domain-randomized-grpo-v2\n- **Demand:** fresh episode seed; new customer draws every retailer/week; demand-responsive retailer orders\n\nPlay locally after \`npm run build\`, or on Cloudflare Pages. Archived v1 model scores use the old counterparty policy and are not compared with v2 human sessions.\n`,
);
console.log(`Built ${PAGES}`);
console.log(`Built ${SPACE}`);
