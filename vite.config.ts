import { createReadStream, statSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Connect } from "vite";
import { defineConfig, type Plugin } from "vite";

/**
 * Demo server (`yarn demo`). The page loads the model from `./model/`, which this config serves from `MODEL_DIR`
 * (default: the repository's exported run, `training/outputs/e5-small/onnx`). A built demo (`yarn demo:build`, in
 * `demo-dist/`) needs the model next to it at `model/` or a `?model=<url>` query parameter.
 */
const MODEL_DIR = resolve(process.env.MODEL_DIR ?? join(import.meta.dirname, "../../training/outputs/e5-small/onnx"));
const MODEL_FILES: Record<string, string> = {
  "classifier.json": "application/json",
  "tokenizer.json": "application/json",
  "tokenizer_config.json": "application/json",
  "model.onnx": "application/octet-stream",
  "model_int8.onnx": "application/octet-stream",
};

/** Mounted at `/model/`, so `request.url` is the file name. */
const serveModel: Connect.NextHandleFunction = (request, response, next) => {
  const file = new URL(request.url ?? "/", "http://localhost").pathname.slice(1);
  const type = MODEL_FILES[file];
  if (type === undefined) return next();
  try {
    const { size } = statSync(join(MODEL_DIR, file));
    response.writeHead(200, { "Content-Type": type, "Content-Length": size, "Cache-Control": "no-cache" });
    createReadStream(join(MODEL_DIR, file)).pipe(response);
  } catch {
    response.statusCode = 404;
    response.end(`${file} not found in ${MODEL_DIR}`);
  }
};

function modelDirectory(): Plugin {
  return {
    name: "serve-model-directory",
    configureServer: server => void server.middlewares.use("/model/", serveModel),
    configurePreviewServer: server => void server.middlewares.use("/model/", serveModel),
  };
}

// Cross-origin isolation lets ONNX Runtime use WebAssembly threads; `credentialless` still loads GitHub avatars.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "credentialless" };

export default defineConfig({
  root: "demo",
  base: "./",
  plugins: [modelDirectory()],
  server: { headers: isolation },
  preview: { headers: isolation },
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
