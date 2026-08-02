import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(HERE, "../..");

export function artifactRows() {
  const files = [
    ["development", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_development/results.json"],
    ["validation", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_validation_controls/results.json"],
  ];
  const rows = [];
  for (const [split, relative] of files) {
    const payload = JSON.parse(readFileSync(resolve(ROOT, relative), "utf8"));
    for (const row of payload.episodes) {
      if (row.scenario_id === "t5-strategic-y-v2") rows.push({ ...row, split });
    }
  }
  return rows.sort((left, right) => {
    const splitOrder = { development: 0, validation: 1 };
    return splitOrder[left.split] - splitOrder[right.split] || left.seed_index - right.seed_index;
  });
}

export function pythonOracle(cases) {
  const env = {
    ...process.env,
    PYTHONPATH: resolve(ROOT, "environments/beer_distribution_game"),
  };
  const result = spawnSync(
    "python3",
    [resolve(ROOT, "static_web/test/python_oracle.py")],
    {
      cwd: ROOT,
      env,
      input: JSON.stringify({ cases }),
      encoding: "utf8",
      maxBuffer: 100 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(`Python oracle failed:\n${result.stderr}`);
  }
  return JSON.parse(result.stdout).cases;
}
