# @hsborges/github-country-classifier

Infer the country of a GitHub user from the public fields of their profile, in Node or in the browser. The model is a
fine-tuned [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) with a linear head
over ISO 3166-1 alpha-2 codes, exported to ONNX (ADRs 0007, 0008 and 0018 of this repository).

It reads only `location`, `company`, `blog`, the domain of the public `email`, `twitter_username` and `bio`. It never
reads the login or the name, nor the local part of the email.

## Usage

The package does not ship the model weights. Pass the directory (Node) or URL (browser) of an exported model:
`classifier.json`, `tokenizer.json`, `tokenizer_config.json`, `model.onnx` (fp32, 470 MB) and/or `model_int8.onnx` (118 MB),
as written by `python -m location_ft.export_onnx`.

```ts
// Node: npm install @hsborges/github-country-classifier onnxruntime-node
import { describeCountryInput, loadCountryClassifier } from "@hsborges/github-country-classifier/node";

const { classify } = await loadCountryClassifier("training/outputs/e5-small/onnx");
const [prediction] = await classify([describeCountryInput({ location: "Recife", bio: "Dev at @acme" })]);
// { country_code: "BR", confidence: 0.99, top: [["BR", 0.99], ["PT", 0.004], ...] }
```

```ts
// Browser: npm install @hsborges/github-country-classifier onnxruntime-web
import { describeCountryInput, loadCountryClassifier } from "@hsborges/github-country-classifier/web";

const { classify } = await loadCountryClassifier("https://example.org/models/e5-small/", {
  onProgress: ({ file, loaded, total }) => console.log(file, loaded, total),
});
```

- `country_code` is null when the calibrated probability is below the threshold of `classifier.json` (or `threshold`);
  `confidence` and `top` are still reported. An input with none of the fields answers null without running the model.
- The browser loader uses `model_int8.onnx` and batches of one by default (dynamic int8 quantization makes an answer depend
  on the other inputs of its batch), keeps the downloaded files in the Cache API, and imports `onnxruntime-web` lazily.
  `wasmPaths` and `executionProviders` are passed to ONNX Runtime.
- The core entry (`@hsborges/github-country-classifier`) has the runtime-independent parts: `describeCountryInput`,
  `createCountryClassifier` (bring your own tokenizer and `RunLogits`), `predictFromLogits`, `parseClassifierConfig`.

### Bundling for the browser

ONNX Runtime starts its WebAssembly threads as module workers of the script that contains it. With Vite, exclude
`onnxruntime-web` from `optimizeDeps` and make sure the chunk holding it does not import your entry chunk (see
`vite.config.ts`); threads also need a cross-origin-isolated page (`Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: credentialless` or `require-corp`). Without isolation it runs single-threaded.

## Limits

Calibrated on users who declare a location. On that test split the model answered 98.9% correctly; on users without a
declared location (checked against an LLM teacher) precision is markedly lower. Treat answers as estimates for aggregate
analyses, not as facts about individuals.

## Demo

`demo/` is a page that reads the latest [public GitHub events](https://docs.github.com/rest/activity/events#list-public-events),
fetches each actor's profile, classifies it in the browser and plots the countries on a world map, with a refresh button,
a minimum-confidence slider and an optional GitHub token (kept in `sessionStorage`; without one GitHub allows 60
requests per hour, one per event list and one per profile). Profiles per refresh default to 20 without a token and to
100 (a whole events page) with one, unless another number was entered.

Profiles are fetched six at a time through Octokit with `@octokit/plugin-throttling` and `@octokit/plugin-retry`:
transient failures are retried twice, a secondary limit of up to 60 s is waited out once, and an exhausted hourly quota
stops the refresh at once (in-flight lookups are aborted) instead of waiting up to an hour. Actors not looked up are
picked up by the next refresh.

```sh
yarn demo                              # from the repository root; http://localhost:5173
MODEL_DIR=/path/to/onnx yarn demo      # another exported model (default: training/outputs/e5-small/onnx)
yarn workspace @hsborges/github-country-classifier demo:build   # static site in demo-dist/
```

The dev and preview servers serve the model at `/model/`. A static deployment needs the model files at `model/` next to
`index.html`, or `?model=<url>` pointing at a CORS-enabled copy; `?onnx=model.onnx` selects the fp32 file.

## Development

`yarn build` (declarations and ESM in `dist/`), `yarn typecheck`, `yarn test`. The runtime tests compare both loaders with
the Python evaluation on `src/fixtures/parity.json` and are skipped when `training/outputs/e5-small/onnx` is absent.
