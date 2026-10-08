import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { createClassifierFromArtifacts, type CountryClassifier, type RunLogits } from "./classifier.js";
import { parseClassifierConfig, type ClassifierConfig } from "./config.js";

export * from "./index.js";

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await readFile(path, "utf8"));
}

export async function readClassifierConfig(directory: string): Promise<ClassifierConfig> {
  return parseClassifierConfig(await readJson(join(directory, "classifier.json")));
}

export interface LoadCountryClassifierOptions {
  /**
   * ONNX file inside the directory (default `model.onnx`, fp32). `model_int8.onnx` is 4x smaller, but its dynamic
   * activation quantization makes an answer depend on the other profiles in the batch: use it with `batchSize: 1`.
   */
  onnxFile?: string | undefined;
  /** Overrides the threshold of `classifier.json`. */
  threshold?: number | undefined;
  /** Most texts per model run (default: all texts of a call at once). */
  batchSize?: number | undefined;
  /** ONNX Runtime intra-op threads (default: its own). */
  threads?: number | undefined;
}

/**
 * Load an exported model directory (`classifier.json`, `tokenizer.json`, `tokenizer_config.json` and the ONNX file)
 * and run it on CPU with `onnxruntime-node`, which is imported here, so importing this module costs nothing.
 */
export async function loadCountryClassifier(directory: string, options: LoadCountryClassifierOptions = {}): Promise<CountryClassifier> {
  const { onnxFile = "model.onnx", threshold, batchSize, threads } = options;
  const [config, tokenizerJson, tokenizerConfig, ort] = await Promise.all([
    readJson(join(directory, "classifier.json")),
    readJson(join(directory, "tokenizer.json")),
    readJson(join(directory, "tokenizer_config.json")),
    import("onnxruntime-node"),
  ]);
  const session = await ort.InferenceSession.create(join(directory, onnxFile), {
    executionProviders: ["cpu"],
    ...(threads === undefined ? {} : { intraOpNumThreads: threads }),
  });
  const run: RunLogits = async (inputIds, attentionMask, batchSize, sequenceLength) => {
    const dims = [batchSize, sequenceLength];
    const output = await session.run({ input_ids: new ort.Tensor("int64", inputIds, dims), attention_mask: new ort.Tensor("int64", attentionMask, dims) });
    const logits = output.logits;
    if (logits === undefined || !(logits.data instanceof Float32Array)) throw new Error("the ONNX model has no float32 logits output");
    return logits.data;
  };
  return createClassifierFromArtifacts({ config, tokenizerJson, tokenizerConfig, run, threshold, batchSize });
}
