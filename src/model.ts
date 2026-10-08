import type { ModelFiles } from "./classifier.js";

/**
 * The model shipped in this package's `model/` directory (int8, trimmed vocabulary; README, "Export"). The URLs are `file:` URLs
 * in Node and emitted assets after a bundler that understands `new URL("…", import.meta.url)` (Vite, webpack 5).
 */
export const bundledModelFiles: ModelFiles = {
  config: new URL("../model/classifier.json", import.meta.url),
  tokenizer: new URL("../model/tokenizer.json", import.meta.url),
  tokenizerConfig: new URL("../model/tokenizer_config.json", import.meta.url),
  model: new URL("../model/model_int8.onnx", import.meta.url),
};
