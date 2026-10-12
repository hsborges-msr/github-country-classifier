import { Tokenizer } from "@huggingface/tokenizers";
import { z } from "zod";
import { parseClassifierConfig, type ClassifierConfig } from "./config.js";

/**
 * A country answer: `country_code` is null below the threshold, and `confidence` is null only for an empty input.
 * The city fields are about the top country: `city` is null unless the country is answered and `city_confidence`
 * reaches the city threshold; `city_confidence` is null when the model has no named city for that country (or no city head).
 * The state fields work the same way with the state head.
 */
export interface CountryPrediction {
  country_code: string | null;
  confidence: number | null;
  /** The three most probable labels with their calibrated probabilities. */
  top: Array<[string, number]>;
  city: string | null;
  /** P(country) x P(city | country) of the first city of `city_top`, never above `confidence`. */
  city_confidence: number | null;
  /** The three most probable named cities of the top country with their probabilities, as `city_confidence`. */
  city_top: Array<[string, number]>;
  state: string | null;
  /** P(country) x P(state | country) of the first state of `state_top`, never above `confidence`. */
  state_confidence: number | null;
  /** The three most probable named states of the top country with their probabilities, as `state_confidence`. */
  state_top: Array<[string, number]>;
}

/** A city or state answer of the top country. */
interface SubAnswer {
  value: string | null;
  confidence: number | null;
  top: Array<[string, number]>;
}

const NO_ANSWER: SubAnswer = Object.freeze({ value: null, confidence: null, top: [] }) as SubAnswer;

export const EMPTY_PREDICTION: CountryPrediction = Object.freeze({
  country_code: null, confidence: null, top: [], city: null, city_confidence: null, city_top: [], state: null, state_confidence: null, state_top: [],
}) as CountryPrediction;

function softmax(logits: readonly number[], temperature: number): number[] {
  const scaled = logits.map(value => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(value => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map(value => value / sum);
}

interface SubHead {
  name: "city" | "state";
  classes: ReadonlyArray<readonly [string, string | null]>;
  temperature: number;
  threshold: number;
}

/** The heads conditional on the country, in the order of their logits after the country ones (ADRs 0023, 0024). */
function subHeads(config: ClassifierConfig): SubHead[] {
  const heads: SubHead[] = [];
  if (config.cities !== undefined) heads.push({ name: "city", classes: config.cities, temperature: config.city_temperature, threshold: config.city_threshold });
  if (config.states !== undefined) heads.push({ name: "state", classes: config.states, temperature: config.state_temperature, threshold: config.state_threshold });
  return heads;
}

/** Width of a model's output row: one logit per label, then one per city class, then one per state class. */
const logitsWidth = (config: ClassifierConfig): number => subHeads(config).reduce((width, head) => width + head.classes.length, config.labels.length);

/** The most probable named class of the `country` among a head's logits, answered when the country is and it reaches the head's threshold. */
function decideSub(head: SubHead, logits: readonly number[], country: [string, number], answered: boolean): SubAnswer {
  const group = head.classes.flatMap(([code], index) => (code === country[0] ? [index] : []));
  const conditional = softmax(group.map(index => logits[index] as number), head.temperature);
  const named = group
    .flatMap((index, k): Array<[string, number]> => {
      const name = head.classes[index]?.[1];
      return name ? [[name, country[1] * (conditional[k] as number)]] : [];
    })
    .sort((a, b) => b[1] - a[1]);
  const [first] = named;
  if (first === undefined) return NO_ANSWER;
  return { value: answered && first[1] >= head.threshold ? first[0] : null, confidence: first[1], top: named.slice(0, 3) };
}

/**
 * Softmax of `logits / temperature` over the labels; the top label is the answer when its probability reaches
 * `threshold` (default: the config's). With a city head (format 2 or 3) and a state head (format 3), their logits
 * follow in that order: a softmax over the top country's classes gives P(city | country) or P(state | country), and the
 * most probable named class is answered as above (ADRs 0023, 0024).
 */
export function predictFromLogits(logits: ArrayLike<number>, config: ClassifierConfig, threshold: number = config.threshold): CountryPrediction {
  const { labels, temperature } = config;
  if (logits.length !== logitsWidth(config)) throw new Error(`expected ${logitsWidth(config)} logits (one per label, city and state class), got ${logits.length}`);
  const values = Array.from(logits);
  const ranked = softmax(values.slice(0, labels.length), temperature).map((p, index): [string, number] => [labels[index] as string, p]).sort((a, b) => b[1] - a[1]);
  const [best] = ranked as [[string, number]];
  const country_code = best[1] >= threshold ? best[0] : null;
  const answers = { city: NO_ANSWER, state: NO_ANSWER };
  let start = labels.length;
  for (const head of subHeads(config)) {
    answers[head.name] = decideSub(head, values.slice(start, start + head.classes.length), best, country_code !== null);
    start += head.classes.length;
  }
  const { city, state } = answers;
  return {
    country_code, confidence: best[1], top: ranked.slice(0, 3),
    city: city.value, city_confidence: city.confidence, city_top: city.top,
    state: state.value, state_confidence: state.confidence, state_top: state.top,
  };
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

/** Runs the model on an encoded batch, returning `[batchSize, labels (+ cities) (+ states)]` logits in row-major order. */
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
 * The inference rule (README, "How it works") over any runtime: empty texts are answered without the model, the others go through
 * the model in batches of at most `batchSize` and {@link predictFromLogits}.
 */
export function createCountryClassifier(options: CountryClassifierOptions): ClassifyTexts {
  const { config, tokenizer, padId, run, threshold, batchSize = Number.POSITIVE_INFINITY } = options;
  if (!(batchSize >= 1)) throw new Error(`batchSize must be at least 1, got ${batchSize}`);
  const width = logitsWidth(config);
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
