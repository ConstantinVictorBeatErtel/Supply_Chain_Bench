import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["static_web/test/**/*.test.js"],
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});
