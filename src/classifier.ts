import { Tokenizer } from "@huggingface/tokenizers";
import { z } from "zod";
import { parseClassifierConfig, type ClassifierConfig } from "./config.js";

/** A country answer: `country_code` is null below the threshold, and `confidence` is null only for an empty input. */
export interface CountryPrediction {
  country_code: string | null;
  confidence: number | null;
  /** The three most probable labels with their calibrated probabilities. */
  top: Array<[string, number]>;
}

export const EMPTY_PREDICTION: CountryPrediction = Object.freeze({ country_code: null, confidence: null, top: [] }) as CountryPrediction;

/** Softmax of `logits / temperature`; the top label is the answer when its probability reaches `threshold` (default: the config's). */
export function predictFromLogits(logits: ArrayLike<number>, config: ClassifierConfig, threshold: number = config.threshold): CountryPrediction {
  const { labels, temperature } = config;
  if (logits.length !== labels.length) throw new Error(`expected ${labels.length} logits (one per label), got ${logits.length}`);
  const scaled = Array.from(logits, value => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(value => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  const ranked = exps.map((value, index): [string, number] => [labels[index] as string, value / sum]).sort((a, b) => b[1] - a[1]);
  const [best] = ranked as [[string, number]];
  return { country_code: best[1] >= threshold ? best[0] : null, confidence: best[1], top: ranked.slice(0, 3) };
}

/** The part of a tokenizer the classifier needs: ids with the model's special tokens. */
export interface TokenizerLike {
  encode(text: string): { ids: number[] };
}

export interface EncodedBatch {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  batchSize: number;
  sequenceLength: number;
}

/**
 * Tokenize `prefix + text` for each text with special tokens and truncate to `maxLength` keeping the final special
 * token, as `transformers` does for one sequence with `truncation=True`; pad to the longest with `padId`.
 */
export function encodeBatch(tokenizer: TokenizerLike, texts: readonly string[], options: { prefix: string; maxLength: number; padId: number }): EncodedBatch {
  const { prefix, maxLength, padId } = options;
  const rows = texts.map(text => {
    const { ids } = tokenizer.encode(prefix + text);
    return ids.length > maxLength ? [...ids.slice(0, maxLength - 1), ids[ids.length - 1] as number] : ids;
  });
  const sequenceLength = Math.max(...rows.map(ids => ids.length));
  const inputIds = new BigInt64Array(rows.length * sequenceLength).fill(BigInt(padId));
  const attentionMask = new BigInt64Array(rows.length * sequenceLength);
  rows.forEach((ids, row) => {
    ids.forEach((id, column) => {
      inputIds[row * sequenceLength + column] = BigInt(id);
      attentionMask[row * sequenceLength + column] = 1n;
    });
  });
  return { inputIds, attentionMask, batchSize: rows.length, sequenceLength };
}

/** Runs the model on an encoded batch, returning `[batchSize, labels]` logits in row-major order. */
export type RunLogits = (inputIds: BigInt64Array, attentionMask: BigInt64Array, batchSize: number, sequenceLength: number) => Promise<Float32Array>;

/** Classifies a batch of input texts (from `describeCountryInput`), in order. */
export type ClassifyTexts = (texts: readonly string[]) => Promise<CountryPrediction[]>;

export interface CountryClassifierOptions {
  config: ClassifierConfig;
  tokenizer: TokenizerLike;
  padId: number;
  run: RunLogits;
  /** Overrides the threshold of the config. */
  threshold?: number | undefined;
  /** Most texts per model run (default: all at once). Use 1 with a dynamically quantized model, whose answers depend on the batch. */
  batchSize?: number | undefined;
}

/**
 * The inference rule of ADR 0008 over any runtime: empty texts are answered without the model, the others go through
 * the model in batches of at most `batchSize` and {@link predictFromLogits}.
 */
export function createCountryClassifier(options: CountryClassifierOptions): ClassifyTexts {
  const { config, tokenizer, padId, run, threshold, batchSize = Number.POSITIVE_INFINITY } = options;
  if (!(batchSize >= 1)) throw new Error(`batchSize must be at least 1, got ${batchSize}`);
  const width = config.labels.length;
  return async texts => {
    const predictions: CountryPrediction[] = texts.map(() => EMPTY_PREDICTION);
    const indexes = texts.flatMap((text, index) => (text.trim() === "" ? [] : [index]));
    for (let start = 0; start < indexes.length; start += batchSize) {
      const chunk = indexes.slice(start, start + batchSize);
      const batch = encodeBatch(tokenizer, chunk.map(index => texts[index] as string), { prefix: config.prefix, maxLength: config.max_length, padId });
      const logits = await run(batch.inputIds, batch.attentionMask, batch.batchSize, batch.sequenceLength);
      if (logits.length !== chunk.length * width) throw new Error(`expected ${chunk.length * width} logits, got ${logits.length}`);
      chunk.forEach((index, row) => {
        predictions[index] = predictFromLogits(logits.subarray(row * width, (row + 1) * width), config, threshold);
      });
    }
    return predictions;
  };
}

/** Where a model's files are: paths or URLs of `classifier.json`, `tokenizer.json`, `tokenizer_config.json` and the ONNX file. */
export interface ModelFiles {
  config: string | URL;
  tokenizer: string | URL;
  tokenizerConfig: string | URL;
  model: string | URL;
}

/** A ready classifier and the config it was built from. */
export interface CountryClassifier {
  config: ClassifierConfig;
  classify: ClassifyTexts;
}

const tokenizerConfigSchema = z.looseObject({ pad_token: z.string().default("<pad>") });

/**
 * Build a classifier from the parsed contents of an exported model directory (`classifier.json`, `tokenizer.json`,
 * `tokenizer_config.json`) and a runtime for its ONNX file. Shared by the Node and browser loaders.
 */
export function createClassifierFromArtifacts(options: {
  config: unknown;
  tokenizerJson: unknown;
  tokenizerConfig: unknown;
  run: RunLogits;
  threshold?: number | undefined;
  batchSize?: number | undefined;
}): CountryClassifier {
  const config = parseClassifierConfig(options.config);
  const tokenizerConfig = tokenizerConfigSchema.parse(options.tokenizerConfig);
  const tokenizerJson = z.record(z.string(), z.unknown()).parse(options.tokenizerJson);
  const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  const padId = tokenizer.token_to_id(tokenizerConfig.pad_token);
  if (padId === undefined) throw new Error(`pad token ${tokenizerConfig.pad_token} is not in the vocabulary`);
  const classify = createCountryClassifier({ config, tokenizer, padId, run: options.run, threshold: options.threshold, batchSize: options.batchSize });
  return { config, classify };
}
