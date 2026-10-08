import { defineConfig } from "vite";

/**
 * Demo (`yarn demo`, `yarn demo:build` into `demo-dist/`). The page loads the package's own model (`model/`), which the
 * build emits as assets; `?model=<url>` loads another copy.
 */

// Cross-origin isolation lets ONNX Runtime use WebAssembly threads; `credentialless` still loads GitHub avatars.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "credentialless" };

export default defineConfig({
  root: "demo",
  base: "./",
  server: { headers: isolation },
  preview: { headers: isolation },
  // Not pre-bundled: ONNX Runtime finds its files with `new URL(…, import.meta.url)` next to its module.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    outDir: "../demo-dist",
    emptyOutDir: true,
    target: "es2022",
    // ONNX Runtime starts its WebAssembly threads as workers of its own chunk, so that chunk must not import the page's
    // entry (where the bundler would otherwise put shared helpers): the workers would run the page.
    rolldownOptions: { output: { codeSplitting: { groups: [{ name: "runtime-helpers", test: /\0rolldown\/runtime/u, priority: 2 }, { name: "ort", test: /onnxruntime-web/u, priority: 1 }] } } },
  },
});
