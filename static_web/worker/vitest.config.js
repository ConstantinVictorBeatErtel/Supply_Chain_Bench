import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { cloudflareTest, readD1Migrations } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

const HERE = dirname(fileURLToPath(import.meta.url));

export default defineConfig(async () => ({
  plugins: [
    cloudflareTest({
      main: resolve(HERE, "src/index.js"),
      miniflare: {
        compatibilityDate: "2026-07-25",
        d1Databases: ["DB"],
        bindings: {
          ALLOWED_ORIGINS: "https://game.example",
          TEST_MIGRATIONS: await readD1Migrations(resolve(HERE, "migrations")),
        },
      },
    }),
  ],
  test: {
    include: ["static_web/worker/**/*.test.js"],
  },
}));
