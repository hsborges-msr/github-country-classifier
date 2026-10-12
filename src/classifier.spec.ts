import { expect, test, vi } from "vitest";
import { createCountryClassifier, EMPTY_PREDICTION, encodeBatch, predictFromLogits } from "./classifier.js";
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

const cityConfig: ClassifierConfig = {
  ...config,
  format: 2,
  labels: ["BR", "US"],
  temperature: 1,
  threshold: 0.5,
  cities: [["BR", "Recife"], ["BR", "Natal"], ["BR", null], ["US", "Austin"], ["US", null]],
  city_temperature: 1,
  city_threshold: 0.4,
};

test("parseClassifierConfig accepts a city head only in format 2, on known labels", () => {
  expect(parseClassifierConfig(cityConfig)).toEqual(cityConfig);
  expect(() => parseClassifierConfig({ ...cityConfig, format: 1 })).toThrow();
  expect(() => parseClassifierConfig({ ...config, format: 2 })).toThrow();
  expect(() => parseClassifierConfig({ ...cityConfig, cities: [["JP", "Tokyo"]] })).toThrow(/cities/);
  expect(() => parseClassifierConfig({ ...cityConfig, city_threshold: 2 })).toThrow();
});

test("predictFromLogits answers the city of the top country with P(country) x P(city | country)", () => {
  // P(BR) = 0.8; P(city | BR) = 0.5 Recife, 0.3 Natal, 0.2 other; the US city logits are ignored.
  const logits = [0.8, 0.2, 0.5, 0.3, 0.2, 100, 100].map(Math.log);
  const prediction = predictFromLogits(logits, cityConfig);
  expect(prediction).toMatchObject({ country_code: "BR", city: "Recife" });
  expect(prediction.city_confidence).toBeCloseTo(0.4, 10);
  expect(prediction.city_top.map(([name]) => name)).toEqual(["Recife", "Natal"]);
  expect(prediction.city_top[1]?.[1]).toBeCloseTo(0.24, 10);
  expect(predictFromLogits(logits, { ...cityConfig, city_threshold: 0.41 }).city).toBeNull();
  // An unanswered country answers no city; its city confidence is still reported.
  const unanswered = predictFromLogits(logits, cityConfig, 0.9);
  expect(unanswered).toMatchObject({ country_code: null, city: null });
  expect(unanswered.city_confidence).toBeCloseTo(0.4, 10);
  // A country without named cities, and a country-only model, answer no city.
  expect(predictFromLogits([0.2, 0.8, 0.5, 0.5].map(Math.log), { ...cityConfig, cities: [["BR", "Recife"], ["BR", null]] }))
    .toMatchObject({ country_code: "US", city: null, city_confidence: null, city_top: [] });
  expect(predictFromLogits([2, 1, 0, -1], config)).toMatchObject({ city: null, city_confidence: null, city_top: [] });
  expect(() => predictFromLogits([1, 2], cityConfig)).toThrow(/expected 7 logits \(one per label, city and state class\)/);
});

const stateConfig: ClassifierConfig = {
  ...cityConfig,
  format: 3,
  cities: [["BR", "Recife"], ["BR", null]],
  states: [["BR", "Pernambuco"], ["BR", null], ["US", "Texas"], ["US", null]],
  state_temperature: 1,
  state_threshold: 0.6,
};

test("parseClassifierConfig accepts a state head only in format 3, with or without a city head", () => {
  expect(parseClassifierConfig(stateConfig)).toEqual(stateConfig);
  const { cities: _cities, city_temperature: _temperature, city_threshold: _threshold, ...stateOnly } = stateConfig;
  expect(parseClassifierConfig(stateOnly)).toEqual(stateOnly);
  expect(() => parseClassifierConfig({ ...stateConfig, format: 2 })).toThrow();
  expect(() => parseClassifierConfig({ ...cityConfig, format: 3 })).toThrow();
  expect(() => parseClassifierConfig({ ...config, states: stateConfig.states, state_temperature: 1, state_threshold: 0.6 })).toThrow();
  expect(() => parseClassifierConfig({ ...stateConfig, states: [["JP", "Tokyo"]] })).toThrow(/states/);
  expect(() => parseClassifierConfig({ ...stateOnly, cities: stateConfig.cities })).toThrow();
});

test("predictFromLogits answers the state of the top country from the logits after the city ones", () => {
  // P(BR) = 0.8; P(Recife | BR) = 0.5; P(Pernambuco | BR) = 0.9; the US state logits are ignored.
  const logits = [0.8, 0.2, 0.5, 0.5, 0.9, 0.1, 100, 100].map(Math.log);
  const prediction = predictFromLogits(logits, stateConfig);
  expect(prediction).toMatchObject({ country_code: "BR", city: "Recife", state: "Pernambuco" });
  expect(prediction.city_confidence).toBeCloseTo(0.4, 10);
  expect(prediction.state_confidence).toBeCloseTo(0.72, 10);
  expect(prediction.state_top.map(([name]) => name)).toEqual(["Pernambuco"]);
  expect(predictFromLogits(logits, { ...stateConfig, state_threshold: 0.73 }).state).toBeNull();
  expect(predictFromLogits(logits, stateConfig, 0.9)).toMatchObject({ country_code: null, state: null });
  // A city-only model answers no state.
  expect(predictFromLogits([0.8, 0.2, 0.5, 0.3, 0.2, 100, 100].map(Math.log), cityConfig)).toMatchObject({ state: null, state_confidence: null, state_top: [] });
  expect(() => predictFromLogits(logits.slice(0, 6), stateConfig)).toThrow(/expected 8 logits/);
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
  expect(predictions[1]).toEqual(EMPTY_PREDICTION);
  expect(run).toHaveBeenCalledTimes(1);
  expect(run.mock.calls[0]?.[2]).toBe(2);
  expect(await classify(["", " "])).toEqual([EMPTY_PREDICTION, EMPTY_PREDICTION]);
  expect(EMPTY_PREDICTION).toEqual({ country_code: null, confidence: null, top: [], city: null, city_confidence: null, city_top: [], state: null, state_confidence: null, state_top: [] });
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

test("createCountryClassifier splits each row's country and city logits", async () => {
  const run = async (_ids: BigInt64Array, _mask: BigInt64Array, batchSize: number) =>
    new Float32Array(Array.from({ length: batchSize }, () => [0.8, 0.2, 0.5, 0.3, 0.2, 100, 100].map(Math.log)).flat());
  const classify = createCountryClassifier({ config: cityConfig, tokenizer, padId: 1, run });
  expect((await classify(["a", "b"])).map(prediction => [prediction.country_code, prediction.city])).toEqual([["BR", "Recife"], ["BR", "Recife"]]);
});

test("createCountryClassifier rejects a model output of the wrong size", async () => {
  const classify = createCountryClassifier({ config, tokenizer, padId: 1, run: async () => new Float32Array(3) });
  await expect(classify(["bio: x"])).rejects.toThrow(/logits/);
});
