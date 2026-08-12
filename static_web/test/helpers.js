import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(HERE, "../..");

export const THOUGHT_TRACE_PATH = "artifacts/public_game_llm_thoughts/traces.json";

/** The recorded comparison run: actions plus the model's own week-by-week notes. */
export function traceArtifact() {
  return JSON.parse(readFileSync(resolve(ROOT, THOUGHT_TRACE_PATH), "utf8"));
}

export function artifactRows() {
  return [...traceArtifact().episodes].sort((left, right) => (
    ({ development: 0, validation: 1 })[left.split]
    - ({ development: 0, validation: 1 })[right.split]
    || left.seed_index - right.seed_index
  ));
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
