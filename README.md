# @hsborges/github-country-classifier

Infer the country of a GitHub user from the public fields of their profile, in Node or in the browser. The package
includes its model (a fine-tuned multilingual sentence encoder, 52 MB) and runs it locally with ONNX Runtime: no service
to call, and no profile text leaves the process or the browser tab.

It reads only `location`, `company`, `blog`, the domain of the public `email`, `twitter_username` and `bio`, never the
login or the name, and answers an ISO 3166-1 alpha-2 code (one of 209 countries) with a confidence.

## Install

The package is distributed as a tarball attached to the
[GitHub releases](https://github.com/hsborges/github-location-inference/releases) of this repository (it is not on the
npm registry). Install the tarball of a release together with the ONNX Runtime for your platform:

```sh
# Node
npm install https://github.com/hsborges/github-location-inference/releases/download/country-classifier-v0.1.0/hsborges-github-country-classifier-0.1.0.tgz onnxruntime-node
# browser (with a bundler)
npm install https://github.com/hsborges/github-location-inference/releases/download/country-classifier-v0.1.0/hsborges-github-country-classifier-0.1.0.tgz onnxruntime-web
```

The tarball is about 37 MB (69 MB unpacked), almost all of it the model. ONNX Runtime is an optional peer dependency:
install the one for the entry you use.

## Usage

```ts
import { describeCountryInput, loadCountryClassifier } from "@hsborges/github-country-classifier/node"; // or "/web"

const { classify } = await loadCountryClassifier();
const profile = await (await fetch("https://api.github.com/users/octocat")).json();
const [prediction] = await classify([describeCountryInput(profile)]);
// { country_code: "US", confidence: 0.9995, top: [["US", 0.9995], ["FR", 0.00003], ["MX", 0.00003]] }
```

- `describeCountryInput` builds the model input from a profile; `classify` takes many inputs and answers in order.
- `country_code` is `null` when the probability is below the model's threshold (override it with
  `loadCountryClassifier(undefined, { threshold })`). The included model's threshold is 0, so it always answers a
  country; filter on `confidence` (see [Accuracy](#accuracy)). `top` has the three most probable countries. A profile
  with none of the six fields answers `null` without running the model.
- Inputs are run one at a time: the int8 model computes its quantization scales per run, so batching would make an
  answer depend on the other inputs of the batch.
- Another model: pass an exported model directory (Node) or its URL (browser) as the first argument (`classifier.json`,
  `tokenizer.json`, `tokenizer_config.json` and `{ onnxFile }`, default `model.onnx` in Node and `model_int8.onnx` in
  the browser), or a `ModelFiles` object with the path or URL of each file.
- The core entry (`@hsborges/github-country-classifier`) has the parts that need no runtime: `describeCountryInput`,
  `createClassifierFromArtifacts` and `createCountryClassifier` (bring your own `RunLogits`), `predictFromLogits`,
  `parseClassifierConfig`.

### In the browser

`loadCountryClassifier` from `/web` downloads the model once (about 69 MB, or less with HTTP compression) and keeps it
in the Cache API; `onProgress` reports the download. The model files are referenced with
`new URL("…", import.meta.url)`, so your bundler emits them as assets (Vite and webpack 5 do).

- **Vite:** exclude `onnxruntime-web` from `optimizeDeps`. ONNX Runtime starts its WebAssembly threads as module workers
  of the chunk that contains it, so that chunk must not import your entry chunk; see this package's
  [`vite.config.ts`](https://github.com/hsborges/github-location-inference/blob/master/packages/country-classifier/vite.config.ts).
- **Threads** need a cross-origin-isolated page (`Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: credentialless` or `require-corp`); without it ONNX Runtime runs single-threaded.
- `wasmPaths` and `executionProviders` are passed to ONNX Runtime.

## How it works

1. **Input.** `describeCountryInput` writes one `field: value` line per non-empty field, in the order `location`,
   `company`, `blog`, `email_domain` (the domain of the public email, never its local part), `twitter_username`, `bio`,
   whitespace collapsed. For example:

   ```text
   location: Recife, Brazil
   company: CESAR
   email_domain: gmail.com
   ```

   The same function built the training data, so the model sees exactly what it was trained on.
2. **Tokens.** `"query: " + text` is tokenized with the SentencePiece tokenizer of `multilingual-e5-small` (with
   [`@huggingface/tokenizers`](https://github.com/huggingface/tokenizers.js), pure JavaScript), special tokens included,
   truncated to 96 tokens keeping the final `</s>`.
3. **Network.** ONNX Runtime runs `model/model_int8.onnx`: the encoder (12 transformer layers, hidden size 384), the
   average of its outputs over the tokens, and a linear layer with one score per country. Inputs `input_ids` and
   `attention_mask` (int64, `[batch, sequence]`), output `logits` (float32, `[batch, 209]`, in the order of `labels`).
4. **Answer.** `softmax(logits / temperature)`; the most probable country is the answer when its probability reaches
   the threshold. `model/classifier.json` holds `prefix`, `max_length`, `input_fields`, `labels`, `temperature`,
   `threshold`, `target_precision` and `vocabulary` (see [Export](#export)).

## How the model was built

### Data

- **Users:** every actor of the public GitHub events published by [GH Archive](https://www.gharchive.org/) for the 360
  hours from 2026-09-01 00:00 to 2026-09-15 09:00 UTC: 2,145,782 accounts, of which 1,965,503 profiles were fetched
  from the GitHub REST API (the other 180,279 no longer existed). 515,750 of them (26%) declare a location.
- **Labels:** each free-text `location` was normalized (emoji, URLs, placeholders such as "Planet Earth" or "Remote" and
  joke places removed; abbreviations such as "NYC" expanded) and geocoded with [Nominatim](https://nominatim.org/) (data
  © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors); low-importance matches that share no word
  with the query were rejected. 464,036 users had their location resolved to a country, which is their label.
- **Splits:** 464,033 labeled examples over 209 countries, split by a hash of the user id: 394,221 for training, 23,294
  for validation, 46,518 for test. The most frequent labels are US (16.5%), IN, CN, BR, DE, GB, FR and CA.
- **The location stays in the input.** It is the main evidence when present, and users with and without one go through
  the same model. So the test split measures mostly how well the model reads a declared location; the teacher check set
  measures the harder case.
- **Teacher check set:** 2,922 users *without* a declared location whose profile states a place elsewhere (company, bio,
  blog, email domain), labeled by a large language model (DeepSeek V4.1 Flash) that had to quote its evidence from the
  profile and was told not to infer from names; only answers whose quote is found in the profile were kept. They are
  never used for training. Such profiles are few: 2.9% of the 100,028 profiles without a location sent to the model.

### Training

- **Network:** [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (12 layers,
  hidden size 384, 118M parameters), mean pooling of the last hidden states over the attention mask, dropout 0.1 and one
  linear layer over the 209 labels. The input prefix `query: ` is the one the base model expects.
- **Fine-tuning:** all encoder layers except the word embeddings, one epoch over the 394,221 training examples, batch 32,
  AdamW with weight decay 0.01, learning rate 5e-5 for the encoder and 1e-3 for the head, linear schedule with 6% warm-up,
  gradient clipping at 1.0, batches of similar length: 12,320 steps, about 4 minutes on one CUDA GPU in bf16.
- **Calibration:** a temperature (1.141) was fitted on 5,000 validation examples by minimum negative log-likelihood. The
  threshold is the lowest value on a 0.01 grid whose validation precision reaches 95%; validation precision was already
  98.8% when answering every example, so the threshold is 0.

### Export

1. **ONNX:** the fine-tuned network was exported with PyTorch's TorchScript exporter (opset 17) and checked against
   PyTorch on test texts.
2. **Vocabulary trimming:** the base model has an embedding row for each of 250,037 token ids, 96M of its 118M
   parameters. Only the tokens that occur in the model inputs of the 1,965,503 fetched profiles were kept: 76,156. The
   tokenizer is unchanged; the graph starts with a lookup from each original id to its row in the smaller table, and ids
   outside it read `<unk>`. On covered texts the trimmed network gives the full network's logits (largest difference
   7.6e-6 over 256 test texts); the Good–Turing estimate of the share of tokens in new profiles never seen is 0.07%.
3. **int8:** ONNX Runtime's dynamic quantization (int8 weights, activations quantized at run time) took the trimmed
   network from 205 MB to 52 MB, against 470 MB for the original fp32 export.

## Accuracy

Measured with ONNX Runtime, one profile per run, on the test split (users who declare a location) and the teacher check
set (users who don't, but state a place elsewhere). "Always answering" counts the most probable country; the other
columns count only answers at or above the minimum confidence.

<!-- ACCURACY-TABLE -->

## Limitations

- **Calibrated on located users.** For users without a location the confidence is optimistic: a 95% minimum confidence
  does not mean 95% precision on them. Most of them state no place at all (see the teacher check set), and a low
  confidence is then the expected answer.
- **Inferred, not stated.** The model guesses from company names, email domains, languages and places in a bio. Use its
  answers for aggregate analyses (where a project's contributors are, say), not as facts about a person, and not to make
  decisions about individuals.
- **A snapshot.** The labels come from the active GitHub users of two weeks of September 2026 and from Nominatim's
  reading of their locations. Countries with few users then were learned from few examples.
- **Fields not used:** the login and the name are excluded on purpose; they correlate with country through names and
  ethnicity, which this model does not use as evidence.

## Demo

`demo/` (in the repository, not in the tarball) reads the latest
[public GitHub events](https://docs.github.com/rest/activity/events#list-public-events), fetches each actor's profile,
classifies it in the browser and plots the countries on a world map, with a refresh button, a minimum-confidence
slider, event counts per user and an optional GitHub token (kept in `sessionStorage`; without one GitHub allows 60
requests per hour, one per event list and one per profile). Profiles per refresh default to 20 without a token and to
100 (a whole events page) with one, unless another number was entered.

Profiles are fetched six at a time through Octokit with `@octokit/plugin-throttling` and `@octokit/plugin-retry`:
transient failures are retried twice, a secondary limit of up to 60 s is waited out once, and an exhausted hourly quota
stops the refresh at once (in-flight lookups are aborted) instead of waiting up to an hour. Actors not looked up are
picked up by the next refresh.

```sh
yarn demo                                                        # from the repository root; http://localhost:5173
yarn workspace @hsborges/github-country-classifier demo:build   # static site in demo-dist/, model included
```

`?model=<url>` loads another exported model directory (CORS-enabled); `?onnx=model.onnx` selects its fp32 file.

## Development

In the repository (the model files are in Git LFS: run `git lfs pull` after cloning):

- `yarn build` (ESM and declarations in `dist/`), `yarn typecheck`, `yarn test`. The tests compare the loaders with the
  Python evaluation on `src/fixtures/parity.json`; the fp32 checks are skipped when `training/outputs/e5-small/onnx`
  is absent.
- `yarn smoke:pack` packs the package, installs the tarball into a new npm project and classifies a profile with both
  entries (needs the npm registry for ONNX Runtime).
- **Updating the model:** export it with `python -m location_ft.export_onnx --trim-vocab …` (see `training/README.md`),
  copy `classifier.json`, `tokenizer.json`, `tokenizer_config.json` and `model_int8.onnx` into `model/`, update this
  README's numbers and bump the version.
- **Releasing:** bump `version`, then push the tag `country-classifier-v<version>`.
  `.github/workflows/release-country-classifier.yml` tests the package, checks the packed tarball and attaches it to a
  GitHub release.

## License

MIT (code and model). The base model, `intfloat/multilingual-e5-small`, is MIT-licensed. The labels were derived from
public GitHub profiles and geocoded with data © OpenStreetMap contributors. Design records in the repository:
`docs/adrs/0007` (task and training), `0008` (artifact and inference rule), `0018` (this package) and `0019` (the
included model).
