import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const pagesPath = resolve(ROOT, "static_web/wrangler.pages.jsonc");
const workerPath = resolve(ROOT, "static_web/worker/wrangler.jsonc");
const workflowPath = resolve(ROOT, ".github/workflows/ci.yml");

const pages = JSON.parse(readFileSync(pagesPath, "utf8"));
const worker = JSON.parse(readFileSync(workerPath, "utf8"));
if (pages.pages_build_output_dir !== "../dist/cloudflare-pages"
    || pages.name !== "beer-distribution-game") {
  throw new Error("invalid Cloudflare Pages configuration");
}
if (worker.main !== "src/index.js"
    || worker.d1_databases?.[0]?.binding !== "DB"
    || worker.d1_databases?.[0]?.migrations_dir !== "migrations") {
  throw new Error("invalid logging Worker configuration");
}
if (!existsSync(resolve(ROOT, "dist/cloudflare-pages/index.html"))
    || !existsSync(resolve(ROOT, "dist/huggingface-space/README.md"))) {
  throw new Error("static build outputs are incomplete");
}
const workflow = readFileSync(workflowPath, "utf8");
for (const required of [
  "python -m pytest -q", "npm run test:all", "npm run check:worker", "npm run build",
]) {
  if (!workflow.includes(required)) throw new Error(`CI workflow is missing: ${required}`);
}
console.log("Validated Pages, Worker, build outputs, and CI command coverage.");
