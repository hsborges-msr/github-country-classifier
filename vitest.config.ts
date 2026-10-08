import { defineConfig } from "vitest/config";

// A separate config so Vitest does not load the demo's vite.config.ts.
export default defineConfig({ test: { include: ["src/**/*.spec.ts", "demo/**/*.spec.ts"], testTimeout: 120_000 } });
