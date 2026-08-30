import {
  existsSync, readFileSync, readdirSync, statSync,
} from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";

const ROOT = resolve(import.meta.dirname, "../..");
const PAGES = resolve(ROOT, "dist/cloudflare-pages");
const SPACE = resolve(ROOT, "dist/huggingface-space");

function relativeFiles(root, current = root) {
  return readdirSync(current).flatMap((name) => {
    const path = resolve(current, name);
    if (statSync(path).isDirectory()) return relativeFiles(root, path);
    return [path.slice(root.length + 1)];
  }).sort();
}

describe("static build", () => {
  test("emits directly loadable Cloudflare Pages and Static Space bundles", () => {
    for (const output of [PAGES, SPACE]) {
      for (const file of [
        "index.html", "app.js", "styles.css", "telemetry.js", "config.js",
        "sim/index.js", "data/llm-comparison.json", "data/benchmark-replay.json",
        "lora-grpo/index.html", "lora-grpo/styles.css", "lora-grpo/og.png",
      ]) {
        expect(existsSync(resolve(output, file)), `${output}/${file}`).toBe(true);
      }
    }
    expect(readFileSync(resolve(SPACE, "README.md"), "utf8")).toContain("sdk: static");
    const pagesFiles = relativeFiles(PAGES);
    const spaceFiles = relativeFiles(SPACE).filter((file) => file !== "README.md");
    expect(spaceFiles).toEqual(pagesFiles);
    for (const file of pagesFiles) {
      expect(readFileSync(resolve(SPACE, file))).toEqual(readFileSync(resolve(PAGES, file)));
    }
  });

  test("uses the canonical repository and GitHub Pages URLs", () => {
    const read = (path) => readFileSync(resolve(ROOT, path), "utf8");
    const publicFiles = [
      read("README.md"),
      read("static_web/public/lora-grpo/index.html"),
      read("dist/cloudflare-pages/lora-grpo/index.html"),
    ].join("\n");

    expect(publicFiles).not.toContain("beer_distribution_RL");
    expect(publicFiles).toContain(
      "https://constantinvictorbeatertel.github.io/Supply_Chain_Bench/lora-grpo/",
    );
    expect(publicFiles).toContain(
      "https://github.com/ConstantinVictorBeatErtel/Supply_Chain_Bench",
    );
  });

  test("generates exactly the recorded comparison catalog", () => {
    const catalog = JSON.parse(readFileSync(
      resolve(PAGES, "data/llm-comparison.json"), "utf8",
    ));
    const artifact = JSON.parse(readFileSync(
      resolve(ROOT, "results/standard/qwen3.5-4b-grpo.json"), "utf8",
    ));
    expect(catalog.environment_version).toBe("live-y-domain-randomized-grpo-v1");
    expect(catalog.scenario_id).toBe("supplychainbench-standard-v1");
    expect(catalog.controlled_role).toBe("wholesaler");
    expect(catalog.capacity).toBe(400);
    expect(catalog.model).toBe("hf:Qwen/Qwen3.5-4B");
    expect(catalog.model_label).toBe("Qwen3.5-4B GRPO");
    expect(catalog.seeds).toHaveLength(16);

    for (const row of catalog.seeds) {
      const recorded = artifact.episodes.find(
        (candidate) => candidate.seed === row.master_seed_hex,
      );
      expect(recorded).toBeTruthy();
      expect(row.actions).toEqual(recorded.actions);
      expect(row.local_total_cost).toBe(recorded.local_total_cost);
      expect(row.scenario.capacity).toBe(400);
      expect(row.scenario.master_seed_hex).toBe(recorded.seed);
    }
  });

  test("ships the trained Qwen cost trace for every benchmark episode", () => {
    const catalog = JSON.parse(readFileSync(
      resolve(PAGES, "data/llm-comparison.json"), "utf8",
    ));
    for (const seed of catalog.seeds) {
      expect(seed.actions).toHaveLength(36);
      expect(seed.costs_over_time).toHaveLength(36);
      expect(seed.costs_over_time.every((cost, index, all) => (
        Number.isFinite(cost) && (index === 0 || cost >= all[index - 1])
      ))).toBe(true);
    }
  });

  test("ships no credential-shaped values", () => {
    const allText = relativeFiles(PAGES)
      .filter((file) => !file.endsWith(".woff2"))
      .map((file) => readFileSync(resolve(PAGES, file), "utf8"))
      .join("\n");
    expect(allText).not.toMatch(/(?:OPENROUTER_API_KEY|AKASH_API_KEY|HF_TOKEN|CLOUDFLARE_API_TOKEN)/);
    expect(allText).not.toMatch(/\b(?:sk|hf)_[A-Za-z0-9_-]{16,}\b/);
  });

  test("ships the deterministic three-policy standard replay", () => {
    const replay = JSON.parse(readFileSync(
      resolve(PAGES, "data/benchmark-replay.json"), "utf8",
    ));
    expect(replay.suite).toBe("standard");
    expect(replay.seed).toMatch(/^[0-9a-f]{16}$/);
    expect(replay.models.map((model) => model.label)).toEqual([
      "adaptive baseline", "untrained Qwen", "trained Qwen",
    ]);
    expect(replay.models.every((model) => model.actions.length === 36)).toBe(true);
    expect(replay.models.every((model) => model.frames.length === 37)).toBe(true);
  });

  test("keeps the public UI seed-opaque and local-only", () => {
    const app = readFileSync(resolve(PAGES, "app.js"), "utf8");
    expect(app).not.toContain("seed-select");
    expect(app).not.toContain("Data collection notice.");
    expect(app).toContain("graphHtml");
    expect(app).toContain('graphHtml("Stock"');
    expect(app).toContain('graphHtml("Flow"');
    expect(app).toContain('graphHtml("Cumulative local cost · operational weeks"');
    expect(app).toContain("trainingScenario");
    expect(app).toContain("Retailer orders");
    expect(app).not.toContain("TRAINED QWEN");
  });
});
