import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, test, vi } from "vitest";
import { HAS_MODEL, MODEL_DIR, readParityCases } from "./fixtures/model.js";
import { loadCountryClassifier } from "./web.js";

/** Serves the model directory as if it were at https://models.test/e5/. */
function serveModelDirectory() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(input instanceof Request ? input.url : input);
    if (url.origin !== "https://models.test" || !url.pathname.startsWith("/e5/")) return new Response(null, { status: 404 });
    try {
      return new Response(await readFile(join(MODEL_DIR, url.pathname.slice("/e5/".length))));
    } catch {
      return new Response(null, { status: 404 });
    }
  });
}

test.skipIf(!HAS_MODEL)("the browser runtime answers as the Python evaluation did and reports progress", async () => {
  const fetch = serveModelDirectory();
  const progress = vi.fn();
  const { classify } = await loadCountryClassifier("https://models.test/e5", { onnxFile: "model.onnx", cache: false, fetch, onProgress: progress });
  expect(fetch.mock.calls.map(([url]) => String(url)).sort()).toEqual(
    ["classifier.json", "model.onnx", "tokenizer.json", "tokenizer_config.json"].map(file => `https://models.test/e5/${file}`),
  );
  expect(progress).toHaveBeenCalledWith(expect.objectContaining({ file: "model.onnx" }));
  const cases = await readParityCases();
  const predictions = await classify(cases.map(item => item.text));
  predictions.forEach((prediction, index) => {
    expect(prediction.top[0]?.[0]).toBe(cases[index]?.country_code);
    expect(prediction.confidence).toBeCloseTo(cases[index]?.confidence ?? Number.NaN, 3);
  });
  expect(await classify([""])).toEqual([{ country_code: null, confidence: null, top: [] }]);
});

test.skipIf(!HAS_MODEL)("the int8 model agrees with fp32 on confident answers", async () => {
  const { classify } = await loadCountryClassifier("https://models.test/e5/", { cache: false, fetch: serveModelDirectory() });
  const cases = (await readParityCases()).filter(item => item.confidence > 0.9);
  const predictions = await classify(cases.map(item => item.text));
  expect(predictions.map(prediction => prediction.country_code)).toEqual(cases.map(item => item.country_code));
});

test.skipIf(!HAS_MODEL)("concurrent calls answer as sequential ones", async () => {
  const { classify } = await loadCountryClassifier("https://models.test/e5/", { cache: false, fetch: serveModelDirectory() });
  const texts = (await readParityCases()).slice(0, 8).map(item => item.text);
  const sequential = await classify(texts);
  const concurrent = await Promise.all(texts.map(async text => (await classify([text]))[0]));
  expect(concurrent).toEqual(sequential);
});

/** Serves `file:` URLs, which Node's fetch does not. */
const fetchFileUrls = vi.fn(async (input: RequestInfo | URL) => new Response(await readFile(new URL(input instanceof Request ? input.url : input))));

test("loads the bundled model by default, which agrees with the Python evaluation on confident answers", async () => {
  const { config, classify } = await loadCountryClassifier(undefined, { cache: false, fetch: fetchFileUrls });
  expect(fetchFileUrls.mock.calls.map(([url]) => String(url).split("/").at(-1)).sort()).toEqual(["classifier.json", "model_int8.onnx", "tokenizer.json", "tokenizer_config.json"]);
  expect(config.labels).toContain("BR");
  const cases = (await readParityCases()).filter(item => item.confidence > 0.9);
  const predictions = await classify(cases.map(item => item.text));
  expect(predictions.map(prediction => prediction.top[0]?.[0])).toEqual(cases.map(item => item.country_code));
});

test.skipIf(!HAS_MODEL)("loads explicit model files", async () => {
  const file = (name: string) => new URL(name, `file://${MODEL_DIR}`);
  const { classify } = await loadCountryClassifier(
    { config: file("classifier.json"), tokenizer: file("tokenizer.json"), tokenizerConfig: file("tokenizer_config.json"), model: file("model.onnx") },
    { cache: false, fetch: fetchFileUrls },
  );
  const [first] = await readParityCases();
  expect((await classify([first?.text ?? ""]))[0]?.confidence).toBeCloseTo(first?.confidence ?? Number.NaN, 3);
});

test("progress ignores the Content-Length of a compressed response, which counts compressed bytes", async () => {
  const progress = vi.fn();
  // Fetch has already decompressed the body, so more bytes arrive than Content-Length announces.
  const fetch = vi.fn(async () => new Response("x".repeat(100), { headers: { "content-encoding": "gzip", "content-length": "10" } }));
  await loadCountryClassifier("https://models.test/e5/", { cache: false, fetch, onProgress: progress }).catch(() => undefined);
  const classifierProgress = progress.mock.calls.map(([event]) => event).filter(event => event.file === "classifier.json");
  expect(classifierProgress.at(-1)).toEqual({ file: "classifier.json", loaded: 100, total: 100 });
  expect(classifierProgress.slice(0, -1).every(event => event.total === null)).toBe(true);
});

test("a missing file fails with its URL", async () => {
  const fetch = vi.fn(async () => new Response(null, { status: 404 }));
  await expect(loadCountryClassifier("https://models.test/none/", { cache: false, fetch })).rejects.toThrow(/could not download https:\/\/models\.test\/none\/\S+: HTTP 404/);
});
