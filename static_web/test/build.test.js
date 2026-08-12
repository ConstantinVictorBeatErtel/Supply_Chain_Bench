import {
  existsSync, readFileSync, readdirSync, statSync,
} from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { traceArtifact } from "./helpers.js";

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

  test("generates exactly the recorded comparison catalog", () => {
    const catalog = JSON.parse(readFileSync(
      resolve(PAGES, "data/llm-comparison.json"), "utf8",
    ));
    const artifact = traceArtifact();
    expect(catalog.environment_version).toBe("0.2.0");
    expect(catalog.scenario_id).toBe("t5-strategic-y-v2");
    expect(catalog.controlled_role).toBe("wholesaler");
    expect(catalog.capacity).toBe(400);
    expect(catalog.model).toBe(artifact.model);
    expect(catalog.seeds).toHaveLength(8);

    for (const row of catalog.seeds) {
      const recorded = artifact.episodes.find(
        (candidate) => candidate.split === row.split && candidate.seed_index === row.seed_index,
      );
      expect(recorded).toMatchObject({ ...row, weeks: expect.anything() });
      expect(row.weeks.map((week) => week.thought))
        .toEqual(recorded.weeks.map((week) => week.thought));
    }
  });

  test("ships the model's week-by-week notes to the client", () => {
    const catalog = JSON.parse(readFileSync(
      resolve(PAGES, "data/llm-comparison.json"), "utf8",
    ));
    for (const seed of catalog.seeds) {
      expect(seed.weeks).toHaveLength(36);
      for (const week of seed.weeks) {
        expect(Object.keys(week).sort()).toEqual([
          "demand", "ending_backlog", "ending_inventory", "quantity", "thought", "week",
        ]);
      }
      // A tracker with nothing to show would be worse than no tracker.
      expect(seed.weeks.filter((week) => week.thought.trim()).length).toBeGreaterThan(30);
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
    expect(app).toContain("graphHtml");
    expect(app).toContain('graphHtml("Stock"');
    expect(app).toContain('graphHtml("Flow"');
  });
});
