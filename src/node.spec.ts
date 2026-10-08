import { expect, test } from "vitest";
import { HAS_MODEL, MODEL_DIR, readParityCases } from "./fixtures/model.js";
import { loadCountryClassifier, readClassifierConfig } from "./node.js";

test("loads the bundled model by default, which agrees with the Python evaluation on confident answers", async () => {
  const { config, classify } = await loadCountryClassifier();
  expect(config.labels).toContain("BR");
  const cases = (await readParityCases()).filter(item => item.confidence > 0.9);
  const predictions = await classify(cases.map(item => item.text));
  expect(predictions.map(prediction => prediction.top[0]?.[0])).toEqual(cases.map(item => item.country_code));
});

test.skipIf(!HAS_MODEL)("readClassifierConfig reads classifier.json from a directory", async () => {
  expect((await readClassifierConfig(MODEL_DIR)).labels).toContain("BR");
});

test.skipIf(!HAS_MODEL)("the fp32 model answers as the Python evaluation did, batched or not", async () => {
  const cases = await readParityCases();
  const texts = cases.map(item => item.text);
  for (const batchSize of [undefined, 1]) {
    const { classify } = await loadCountryClassifier(MODEL_DIR, { batchSize });
    const predictions = await classify(texts);
    predictions.forEach((prediction, index) => {
      expect(prediction.top[0]?.[0]).toBe(cases[index]?.country_code);
      expect(prediction.confidence).toBeCloseTo(cases[index]?.confidence ?? Number.NaN, 3);
    });
  }
});
