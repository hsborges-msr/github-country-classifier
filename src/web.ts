import type { InferenceSession } from "onnxruntime-web";
import { createClassifierFromArtifacts, type CountryClassifier, type RunLogits } from "./classifier.js";

export * from "./index.js";

/** Download progress of one model file; `total` is null when the server sends no length. */
export interface LoadProgress {
  file: string;
  loaded: number;
  total: number | null;
}

export interface LoadCountryClassifierOptions {
  /**
   * ONNX file under the base URL (default `model_int8.onnx`, 4x smaller than the fp32 `model.onnx`). Its dynamic
   * activation quantization depends on the batch, so the default `batchSize` is 1.
   */
  onnxFile?: string | undefined;
  /** Overrides the threshold of `classifier.json`. */
  threshold?: number | undefined;
  /** Most texts per model run (default 1, so every answer depends only on its own text). */
  batchSize?: number | undefined;
  /** Cache API cache that keeps downloaded files across visits (default `github-country-classifier`); `false` disables it. */
  cache?: string | false | undefined;
  /** Called while files download. */
  onProgress?: ((progress: LoadProgress) => void) | undefined;
  /** Where ONNX Runtime fetches its `.wasm` files from (sets `ort.env.wasm.wasmPaths`); the default is next to its script. */
  wasmPaths?: string | undefined;
  /** ONNX Runtime execution providers (default `["wasm"]`). */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];
  /** Replaces the global `fetch`, for example in tests. */
  fetch?: typeof fetch | undefined;
}

const DEFAULT_CACHE = "github-country-classifier";

function resolveBase(baseUrl: string | URL): URL {
  const base = new URL(baseUrl, globalThis.location?.href);
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  return base;
}

async function openCache(name: string | false): Promise<Cache | null> {
  if (name === false || typeof caches === "undefined") return null;
  try {
    return await caches.open(name);
  } catch {
    return null;
  }
}

async function readBody(response: Response, file: string, onProgress: LoadCountryClassifierOptions["onProgress"]): Promise<Uint8Array> {
  const length = Number(response.headers.get("content-length"));
  const total = Number.isFinite(length) && length > 0 ? length : null;
  if (response.body === null || onProgress === undefined) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress?.({ file, loaded: bytes.length, total: bytes.length });
    return bytes;
  }
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress({ file, loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  onProgress({ file, loaded, total: loaded });
  return bytes;
}

/**
 * Load an exported model directory served at `baseUrl` (`classifier.json`, `tokenizer.json`, `tokenizer_config.json`
 * and the ONNX file) and run it with `onnxruntime-web`, which is imported here. Files are kept in the Cache API, so later visits skip the download.
 */
export async function loadCountryClassifier(baseUrl: string | URL, options: LoadCountryClassifierOptions = {}): Promise<CountryClassifier> {
  const { onnxFile = "model_int8.onnx", threshold, batchSize = 1, cache: cacheName = DEFAULT_CACHE, onProgress, wasmPaths, executionProviders = ["wasm"] } = options;
  const fetchFile = options.fetch ?? globalThis.fetch.bind(globalThis);
  // Imported here so bundlers emit ONNX Runtime as its own chunk: its WebAssembly threads start workers from the URL
  // of the script that contains it, which must not be the page's own script.
  const ort = await import("onnxruntime-web");
  if (wasmPaths !== undefined) ort.env.wasm.wasmPaths = wasmPaths;
  const base = resolveBase(baseUrl);
  const cache = await openCache(cacheName);

  const download = async (file: string): Promise<Uint8Array> => {
    const url = new URL(file, base).href;
    const cached = await cache?.match(url);
    if (cached !== undefined) return readBody(cached, file, onProgress);
    const response = await fetchFile(url);
    if (!response.ok) throw new Error(`could not download ${url}: HTTP ${response.status}`);
    if (cache !== null) await cache.put(url, response.clone()).catch(() => undefined);
    return readBody(response, file, onProgress);
  };
  const json = async (file: string): Promise<unknown> => JSON.parse(new TextDecoder().decode(await download(file)));

  const [config, tokenizerJson, tokenizerConfig, model] = await Promise.all([json("classifier.json"), json("tokenizer.json"), json("tokenizer_config.json"), download(onnxFile)]);
  const session = await ort.InferenceSession.create(model, { executionProviders });
  const run: RunLogits = async (inputIds, attentionMask, batchSize, sequenceLength) => {
    const dims = [batchSize, sequenceLength];
    const output = await session.run({ input_ids: new ort.Tensor("int64", inputIds, dims), attention_mask: new ort.Tensor("int64", attentionMask, dims) });
    const logits = output.logits;
    if (logits === undefined || !(logits.data instanceof Float32Array)) throw new Error("the ONNX model has no float32 logits output");
    return logits.data;
  };
  return createClassifierFromArtifacts({ config, tokenizerJson, tokenizerConfig, run, threshold, batchSize });
}
