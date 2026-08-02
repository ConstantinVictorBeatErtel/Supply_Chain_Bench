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
  const sources = [
    ["development", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_development/results.json"],
    ["validation", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_validation_controls/results.json"],
  ];
  const seeds = [];
  for (const [split, path] of sources) {
    for (const episode of loadJson(path).episodes) {
      if (episode.scenario_id !== "t5-strategic-y-v2") continue;
      seeds.push({
        id: `${split}-${episode.seed_index}`,
        split,
        seed_index: episode.seed_index,
        master_seed_hex: episode.master_seed_hex,
        episode_id: episode.episode_id,
        actions: episode.actions,
        local_total_cost: episode.local_total_cost,
        paired_base_stock_local_total_cost: episode.paired_base_stock_local_total_cost,
        reward: episode.reward,
      });
    }
  }
  seeds.sort((left, right) => {
    const order = { development: 0, validation: 1 };
    return order[left.split] - order[right.split] || left.seed_index - right.seed_index;
  });
  if (seeds.length !== 8) throw new Error(`expected 8 headline LLM traces, found ${seeds.length}`);
  return {
    schema_version: "1.0.0",
    environment_version: "0.2.0",
    scenario_id: "t5-strategic-y-v2",
    controlled_role: "wholesaler",
    model: "LLM",
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
  `---\ntitle: Beer Distribution Game — Human Baseline\nemoji: 🍺\ncolorFrom: green\ncolorTo: yellow\nsdk: static\napp_file: index.html\npinned: false\n---\n\n# Beer Distribution Game — Human Baseline\n\nStatic environment v0.2.0 human-baseline app.\n`,
);
console.log(`Built ${PAGES}`);
console.log(`Built ${SPACE}`);
