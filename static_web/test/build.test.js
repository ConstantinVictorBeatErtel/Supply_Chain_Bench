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
        "sim/index.js", "data/llm-comparison.json",
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

  test("generates exactly the immutable eight-seed LLM catalog", () => {
    const catalog = JSON.parse(readFileSync(
      resolve(PAGES, "data/llm-comparison.json"), "utf8",
    ));
    expect(catalog.environment_version).toBe("0.2.0");
    expect(catalog.scenario_id).toBe("t5-strategic-y-v2");
    expect(catalog.controlled_role).toBe("wholesaler");
    expect(catalog.seeds).toHaveLength(8);

    const sources = [
      ["development", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_development/results.json"],
      ["validation", "artifacts/hub_llm/deepseek_v4_flash/v0_2_wholesaler_y_validation_controls/results.json"],
    ];
    const expected = [];
    for (const [split, path] of sources) {
      const artifact = JSON.parse(readFileSync(resolve(ROOT, path), "utf8"));
      for (const episode of artifact.episodes) {
        if (episode.scenario_id === "t5-strategic-y-v2") {
          expected.push({
            split,
            seed_index: episode.seed_index,
            episode_id: episode.episode_id,
            actions: episode.actions,
            local_total_cost: episode.local_total_cost,
            paired_base_stock_local_total_cost: episode.paired_base_stock_local_total_cost,
            reward: episode.reward,
          });
        }
      }
    }
    for (const row of catalog.seeds) {
      expect(row).toMatchObject(expected.find(
        (candidate) => candidate.split === row.split && candidate.seed_index === row.seed_index,
      ));
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

  test("keeps the public UI seed-opaque and local-only", () => {
    const app = readFileSync(resolve(PAGES, "app.js"), "utf8");
    expect(app).not.toContain("seed-select");
    expect(app).not.toContain("Data collection notice.");
    expect(app).toContain("trajectoryCharts");
    expect(app).toContain("Observed weeks only");
  });
});
