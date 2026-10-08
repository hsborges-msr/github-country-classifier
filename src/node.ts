import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createClassifierFromArtifacts, type CountryClassifier, type ModelFiles, type RunLogits } from "./classifier.js";
import { parseClassifierConfig, type ClassifierConfig } from "./config.js";

export * from "./index.js";

const toPath = (file: string | URL): string => (typeof file === "string" ? file : fileURLToPath(file));

async function readJson(file: string | URL): Promise<unknown> {
  return JSON.parse(await readFile(toPath(file), "utf8"));
}

export async function readClassifierConfig(directory: string): Promise<ClassifierConfig> {
  return parseClassifierConfig(await readJson(join(directory, "classifier.json")));
}

export interface LoadCountryClassifierOptions {
  /**
   * ONNX file inside a model directory (default `model.onnx`, fp32). `model_int8.onnx` is 4x smaller, but its dynamic
   * activation quantization makes an answer depend on the other profiles in the batch: use it with `batchSize: 1`.
   */
  onnxFile?: string | undefined;
  /** Overrides the threshold of `classifier.json`. */
  threshold?: number | undefined;
  /** Most texts per model run (default: 1 for the bundled int8 model, otherwise all texts of a call at once). */
  batchSize?: number | undefined;
  /** ONNX Runtime intra-op threads (default: its own). */
  threads?: number | undefined;
}

async function bundledModelFiles(): Promise<ModelFiles> {
  return (await import("./model.js")).bundledModelFiles;
}

/**
 * Load a model and run it on CPU with `onnxruntime-node`, which is imported here, so importing this module costs nothing.
 * Without `source`, the model in this package (int8, trimmed vocabulary) is used,
 * one text per run; a directory is an exported model directory (`classifier.json`, `tokenizer.json`,
 * `tokenizer_config.json` and `onnxFile`); {@link ModelFiles} names each file.
 */
export async function loadCountryClassifier(source?: string | ModelFiles, options: LoadCountryClassifierOptions = {}): Promise<CountryClassifier> {
  const { onnxFile = "model.onnx", threshold, threads } = options;
  const files: ModelFiles =
    source === undefined
      ? await bundledModelFiles()
      : typeof source === "string"
        ? { config: join(source, "classifier.json"), tokenizer: join(source, "tokenizer.json"), tokenizerConfig: join(source, "tokenizer_config.json"), model: join(source, onnxFile) }
        : source;
  const batchSize = options.batchSize ?? (source === undefined ? 1 : undefined);
  const [config, tokenizerJson, tokenizerConfig, ort] = await Promise.all([readJson(files.config), readJson(files.tokenizer), readJson(files.tokenizerConfig), import("onnxruntime-node")]);
  const session = await ort.InferenceSession.create(toPath(files.model), {
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
