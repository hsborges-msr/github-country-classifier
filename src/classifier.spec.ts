import { expect, test, vi } from "vitest";
import { createCountryClassifier, encodeBatch, predictFromLogits } from "./classifier.js";
import { parseClassifierConfig, type ClassifierConfig } from "./config.js";

const config: ClassifierConfig = {
  format: 1,
  base_model: "intfloat/multilingual-e5-small",
  prefix: "query: ",
  max_length: 6,
  pooling: "mean",
  input_fields: ["location", "company", "blog", "email_domain", "twitter_username", "bio"],
  labels: ["BR", "DE", "US", "PT"],
  temperature: 0.5,
  threshold: 0.6,
  target_precision: 0.95,
};

test("parseClassifierConfig accepts the contract and rejects another input or pooling", () => {
  expect(parseClassifierConfig({ ...config, extra: true })).toEqual(config);
  expect(() => parseClassifierConfig({ ...config, input_fields: ["login", "bio"] })).toThrow(/input fields/);
  expect(() => parseClassifierConfig({ ...config, pooling: "cls" })).toThrow();
  expect(() => parseClassifierConfig({ ...config, labels: ["br"] })).toThrow();
  expect(() => parseClassifierConfig({ ...config, format: 2 })).toThrow();
});

test("predictFromLogits applies the temperature, the threshold and reports the top three", () => {
  // logits / 0.5 = [4, 2, 0, -2]
  const prediction = predictFromLogits([2, 1, 0, -1], config);
  const expected = Math.exp(4) / (Math.exp(4) + Math.exp(2) + Math.exp(0) + Math.exp(-2));
  expect(prediction.country_code).toBe("BR");
  expect(prediction.confidence).toBeCloseTo(expected, 10);
  expect(prediction.top.map(([code]) => code)).toEqual(["BR", "DE", "US"]);
  expect(prediction.top.reduce((sum, [, p]) => sum + p, 0)).toBeLessThan(1);
  expect(predictFromLogits([2, 1, 0, -1], config, 0.9)).toMatchObject({ country_code: null, confidence: expected });
  expect(predictFromLogits([1000, 999, 0, 0], config).confidence).toBeCloseTo(1 / (1 + Math.exp(-2)), 10);
  expect(() => predictFromLogits([1, 2], config)).toThrow(/one per label/);
});

const tokenizer = {
  encode: (text: string) => ({ ids: [0, ...[...text.replace("query: ", "")].map(char => char.charCodeAt(0)), 2] }),
};

test("encodeBatch prefixes, truncates keeping the final special token, and pads", () => {
  const batch = encodeBatch(tokenizer, ["ab", "abcdefgh"], { prefix: "query: ", maxLength: 6, padId: 1 });
  expect(batch.sequenceLength).toBe(6);
  expect([...batch.inputIds].map(Number)).toEqual([0, 97, 98, 2, 1, 1, 0, 97, 98, 99, 100, 2]);
  expect([...batch.attentionMask].map(Number)).toEqual([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]);
});

/** A fake model that puts its mass on DE when the first input character is "B", else on US. */
function fakeRun() {
  return vi.fn(async (inputIds: BigInt64Array, _mask: BigInt64Array, batchSize: number, sequenceLength: number) => {
    const logits = new Float32Array(batchSize * config.labels.length);
    for (let row = 0; row < batchSize; row++) {
      const first = Number(inputIds[row * sequenceLength + 1]);
      logits.set(first === "B".charCodeAt(0) ? [0, 5, 0, 0] : [0, 0, 5, 0], row * config.labels.length);
    }
    return logits;
  });
}

test("createCountryClassifier answers empty texts without the model and keeps the input order", async () => {
  const run = fakeRun();
  const classify = createCountryClassifier({ config, tokenizer, padId: 1, run });
  const predictions = await classify(["Berlin", "", "Austin"]);
  expect(predictions.map(prediction => prediction.country_code)).toEqual(["DE", null, "US"]);
  expect(predictions[1]).toEqual({ country_code: null, confidence: null, top: [] });
  expect(run).toHaveBeenCalledTimes(1);
  expect(run.mock.calls[0]?.[2]).toBe(2);
  expect(await classify(["", " "])).toEqual([{ country_code: null, confidence: null, top: [] }, { country_code: null, confidence: null, top: [] }]);
  expect(run).toHaveBeenCalledTimes(1);
});

test("createCountryClassifier runs the model in batches of at most batchSize, in order", async () => {
  const run = fakeRun();
  const classify = createCountryClassifier({ config, tokenizer, padId: 1, run, batchSize: 2 });
  const predictions = await classify(["Berlin", "Austin", "", "Bonn", "Boston"]);
  expect(predictions.map(prediction => prediction.country_code)).toEqual(["DE", "US", null, "DE", "DE"]);
  expect(run.mock.calls.map(call => call[2])).toEqual([2, 2]);
  expect(() => createCountryClassifier({ config, tokenizer, padId: 1, run, batchSize: 0 })).toThrow(/batchSize/);
});

test("createCountryClassifier rejects a model output of the wrong size", async () => {
  const classify = createCountryClassifier({ config, tokenizer, padId: 1, run: async () => new Float32Array(3) });
  await expect(classify(["bio: x"])).rejects.toThrow(/logits/);
});
