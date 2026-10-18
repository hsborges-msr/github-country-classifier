<div align="center">

# GitHub Country Classifier

**Where is a developer from? A compact, calibrated model that reads a GitHub profile and answers the country,
offline, in Node or in the browser.**

[![CI](https://github.com/hsborges-msr/github-country-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/hsborges-msr/github-country-classifier/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/hsborges-msr/github-country-classifier)](https://github.com/hsborges-msr/github-country-classifier/releases)
[![Live demo](https://img.shields.io/badge/demo-live-2ea44f)](https://hsborges-msr.github.io/github-country-classifier/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**[Try the live demo →](https://hsborges-msr.github.io/github-country-classifier/)** It reads GitHub's latest public
events, classifies each author in your browser and puts them on a world map.

<a href="https://hsborges-msr.github.io/github-country-classifier/"><img src="docs/demo.png" width="800" alt="The live demo after one refresh: a world map shaded by the number of users per country, a ranking led by the United States, Germany and India, and totals for all 86 classified users and for the 23 placed at 95% confidence or more"></a>

</div>

Only about one in five active GitHub users fills in the `location` field (22% in our sample of 5.79M), and when they
do it is free text: "SF Bay Area",
"Planet Earth", "深圳", "🇧🇷". Yet the rest of a profile often says where someone is: an employer, a `.de` email domain,
a bio in Portuguese. This package turns those signals into an ISO country code with a calibrated confidence, and the
city and state when the profile names them, using a fine-tuned multilingual transformer that ships inside the package and runs on
your machine.

| | |
| --- | --- |
| 🎯 **Accurate** | **99.0%** on 113,269 held-out users who declare a location (95% CI 99.0–99.1%), across **245 countries** |
| 🔍 **Beyond the location field** | **93.9%** on held-out users with *no* declared location whose profile names a place elsewhere |
| 🏙️ **Cities and states too** | The city among **1,025** frequent ones and the state among **701**, each with its own confidence: **99.99%** precision for half of the users who declare a location |
| 🤐 **Abstains when unsure** | By default answers only at ≥ 94% confidence, a threshold set for 95% precision on profiles without their location: `null` instead of a guess |
| ⚡ **Fast and local** | ~**15 ms per profile** on a 2015 laptop CPU; no API calls, no profile text leaves your process or browser tab |
| 📦 **Self-contained** | The 52 MB model is in the package: one `npm install`, no downloads at run time, no service to host |
| 🌍 **Multilingual** | Built on `multilingual-e5-small`; reads places and bios in Latin, Cyrillic, Chinese, Japanese and Korean script alike |
| 🛡️ **Privacy by design** | Never reads the login or the name, nor the local part of an email |

```ts
const { classify } = await loadCountryClassifier();
await classify([describeCountryInput({ location: "Porto Alegre, RS", bio: "Desenvolvedor backend" })]);
// [{ country_code: "BR", confidence: 0.9996, top: [["BR", 0.9996], ...], city: "Porto Alegre", city_confidence: 0.999, city_top: [...],
//    state: "Rio Grande do Sul", state_confidence: 0.9993, state_top: [...] }]
```

## Quick start

The package is published as a tarball on this repository's
[releases](https://github.com/hsborges-msr/github-country-classifier/releases) (not on the npm registry). Install it with
the ONNX Runtime for your platform:

```sh
# Node
npm install https://github.com/hsborges-msr/github-country-classifier/releases/download/v0.3.0/hsborges-msr-github-country-classifier-0.3.0.tgz onnxruntime-node
# Browser (with a bundler such as Vite or webpack 5)
npm install https://github.com/hsborges-msr/github-country-classifier/releases/download/v0.3.0/hsborges-msr-github-country-classifier-0.3.0.tgz onnxruntime-web
```

```ts
import { describeCountryInput, loadCountryClassifier } from "@hsborges-msr/github-country-classifier/node"; // or "/web"

const { classify } = await loadCountryClassifier();
const profile = await (await fetch("https://api.github.com/users/octocat")).json();
const [prediction] = await classify([describeCountryInput(profile)]);
// { country_code: "US", confidence: 0.9999, top: [["US", 0.9999], ["CN", 0.00003], ["CA", 0.00001]],
//   city: "San Francisco", city_confidence: 0.9997, city_top: [["San Francisco", 0.9997], ["San Jose", 0.00003], ["Palo Alto", 0.00003]],
//   state: "California", state_confidence: 0.9999, state_top: [["California", 0.9999], ["Michigan", 0.00001], ["New York", 0.00001]] }
```

The tarball is 37 MB (70 MB unpacked), almost all of it the model; ONNX Runtime is an optional peer dependency, so
install only the one you use.

## Results

Evaluated with ONNX Runtime on the shipped model file, one profile per run, as the package runs it (the JavaScript
runtimes are tested to give the same answers as the Python evaluation). None of these users was used for training or
calibration. Intervals are Wilson 95% intervals.

| Evaluation set | Majority class | Always answering | ≥ 50% confidence | ≥ 90% | Default threshold (94%) |
| --- | --- | --- | --- | --- | --- |
| **Test split**: 113,269 users who declare a location | 15.4% (IN) | **99.0%** (99.0–99.1) | 99.5% precision, 99.2% answered | 99.9% / 97.8% | **99.9%** (99.9–99.9) / 97.3% |
| **Location hidden**: the same users with the `location` line removed (11% are left with nothing) | 15.4% (IN) | 45.9% (45.6–46.2) | 85.2% / 39.0% | 95.1% / 24.1% | **95.9%** (95.6–96.1) / 21.3% |
| **Teacher test set**: 916 users without a location who name a place elsewhere | 16.8% (US) | **93.9%** (92.1–95.3) | 97.9% / 93.9% | 99.7% / 84.8% | **99.7%** (99.0–99.9) / 81.5% |
| **Domain test set**: 6,935 users without a location whose email or blog domain names a country | 28.6% (CN) | 99.3% (99.1–99.5) | 99.6% / 99.5% | 99.8% / 98.2% | 99.8% (99.7–99.9) / 97.3% |

*Precision* counts correct answers among those at or above the minimum confidence; *answered* is the share of profiles
that reach it. The default threshold is the lowest confidence at which validation precision reaches 95% both for users
with their location and for the same kind of users with it removed, the harder case, which sets it. Pass a lower
`threshold` to answer more profiles at lower precision.

**Cities and states.** On the test split, the model names a city at the default city threshold (99.4%) for 50.8% of
the users, with 99.99% precision (99.97–99.99), and a state at the default state threshold (99.7%) for 51.7%, with
99.99% precision (99.98–99.99). The ceilings are 56.0% for cities, since 36% of these users' locations name no city
(only a state or a country) and the rest name a city outside the 1,025 known ones, and 72.0% for states (26% name only
a country). Without the `location` line it almost never answers either (0.02% and 0.01% of the location-hidden users),
so cities and states come from the location, rarely from the other fields. The teacher and domain test sets have no
city or state labels.

**Compression is almost free in accuracy.** The shipped model is 9× smaller than the full-precision export and loses
0.1–1.3 points when always answering. It is slightly less confident at the top, so at the default thresholds it
answers fewer profiles, with about the same precision:

| Model | Size | Test split | Teacher test set | Location hidden, at the default threshold | Cities, test split | States, test split |
| --- | --- | --- | --- | --- | --- | --- |
| Full fp32 export | 474 MB | 99.1% | 95.2% | 95.5% precision, 22.8% answered | 99.98% / 53.0% | 99.99% / 56.3% |
| **Trimmed vocabulary, int8, shipped** | **52 MB** | **99.0%** | **93.9%** | **95.9%**, 21.3% | **99.99%** / 50.8% | **99.99%** / 51.7% |

**Context from the pilot experiments** (smaller samples and earlier dataset versions, so indicative only): a character
n-gram TF-IDF model with logistic regression reached 88.2% on located users and 77.4% on teacher-labelled unlocated
users, and a generative 0.5B-parameter model needed about 1.2 s per profile on the same laptop CPU, about 80 times
slower than this package.

### What the numbers mean

- **The test split is mostly about reading a declared location.** Those users keep their `location` line in the input,
  as they would at inference. 99.0% is therefore the model's skill at turning messy free text ("NYC", "Bengaluru,
  Karnataka", "Москва") into a country, plus whatever the other fields add.
- **Location hidden is the pessimistic case.** Its label is the country of the declared location, but the remaining
  fields may say nothing (`email_domain: gmail.com`) or point elsewhere (a foreign employer). It is the set the default
  threshold is calibrated for.
- **The teacher and domain test sets are the optimistic case**: profiles with no location that name a place, labelled
  by a large language model that had to quote its evidence from the profile, or by rules on the email and blog domain.
  Such profiles are a minority: 2.7% of the 335,353 location-less profiles sent to the language model named a place at
  all. Agreement with the domain rules is partly by construction, since the model also learned from them.
- **Labels are geocoder, rule and LLM outputs, not ground truth.** Accuracy is agreement with those labels; all make
  mistakes, so part of the residual error is label noise.

## How it works

```mermaid
flowchart LR
  P["GitHub profile"] --> I["describeCountryInput<br/>6 public fields"]
  I --> T["SentencePiece tokenizer<br/>'query: ' + text, ≤ 96 tokens"]
  T --> E["multilingual-e5-small<br/>12 layers · int8 · trimmed vocabulary"]
  E --> H["mean pooling +<br/>linear heads (countries,<br/>cities, states)"]
  H --> C["softmax(logits / T)<br/>calibrated probabilities"]
  C --> A["country_code, city, state<br/>(or null below their thresholds),<br/>confidences, top 3"]
```

1. **Input.** `describeCountryInput` writes one `field: value` line per non-empty field, in the order `location`,
   `company`, `blog`, `email_domain` (the domain of the public email only), `twitter_username`, `bio`, whitespace
   collapsed. The same function built the training data, so the model sees exactly what it was trained on.

   ```text
   location: Recife, Brazil
   company: CESAR
   email_domain: gmail.com
   ```

2. **Tokens.** `"query: " + text` goes through the SentencePiece tokenizer of `multilingual-e5-small`
   ([`@huggingface/tokenizers`](https://github.com/huggingface/tokenizers.js), pure JavaScript), truncated to 96
   tokens.
3. **Network.** ONNX Runtime runs `model/model_int8.onnx`: the encoder (12 transformer layers, hidden size 384), the
   average of its outputs over the tokens, and a linear layer with one score per country, city class and state class.
4. **Answer.** `softmax(logits / temperature)`; the most probable country when its probability reaches the threshold
   (`null` otherwise), its probability and the top three. A profile with none of the six fields answers `null` without
   running the model.
5. **City (models with a city head only).** A `classifier.json` in format 2 or 3 also lists city classes (`[country, name]`,
   with one `[country, null]` "other city" class per country). The model then returns those city logits after the
   country ones. A softmax over the city classes of the top country, at `city_temperature`, gives P(city | country).
   `city_confidence` is P(country) × P(city | country) for the most probable named city, so it is never higher than
   `confidence`. `city` is that name when the country is answered and `city_confidence` reaches `city_threshold`.
   The included model knows 1,025 cities in 123 countries, those with at least 50 training users, with
   `city_threshold` 0.994.
6. **State (models with a state head only).** A `classifier.json` in format 3 lists state classes (`states`, the same
   pairs) with `state_temperature` and `state_threshold`, and the city fields when it also has a city head. The state
   logits come after the city ones, and `state`, `state_confidence` and `state_top` follow the city rule. City and state
   are each conditional on the country only, so they can disagree. The included model knows 701 states (or regions,
   provinces) in 127 countries, with `state_threshold` 0.997; a city that is its own first-level division (Berlin,
   Tokyo, Moscow) is its own state.

## How the model was built

```mermaid
flowchart LR
  G["GH Archive<br/>1,464 hours, Aug–Sep 2026"] --> U["5,787,635<br/>profiles fetched"]
  U --> L["1,252,381 declare<br/>a location"]
  L --> N["Normalize + geocode<br/>(Nominatim / OSM)"]
  N --> D["Labels: 1,132,868 geocoded,<br/>+ domain rules, LLM teacher,<br/>GeoNames cities"]
  D --> F["Fine-tune<br/>multilingual-e5-small"]
  F --> K["Temperature +<br/>threshold"]
  K --> X["Trim vocabulary +<br/>int8 → 52 MB"]
```

### Data

- **Users:** every actor of the public GitHub events published by [GH Archive](https://www.gharchive.org/) for the 1,464
  hours from 2026-08-01 00:00 to 2026-09-30 23:00 UTC: 6,455,397 accounts, of which 5,787,635 profiles were fetched
  from the GitHub REST API (the other 667,762 no longer existed). 1,252,381 of them (22%) declare a location.
- **Labels:** each free-text `location` was normalized (emoji, URLs, placeholders such as "Planet Earth" or "Remote" and
  joke places removed; abbreviations such as "NYC" expanded) and geocoded with [Nominatim](https://nominatim.org/) (data
  © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors); low-importance matches sharing no word with
  the query were rejected. 1,132,868 users had their location resolved to a country, which is their label, 724,504 of them (64%) to a
  city and 838,185 (74%) to a state (the geocoded state, or the city for a city that is its own state, such as Berlin).
- **Users without a location**, two more label sources: 68,599 whose email or blog domain names a country (a
  country-code ending such as `.de`, or a host such as `qq.com` learned from located users), and 9,183 labelled by
  DeepSeek V4.1 Flash with grounded, quoted evidence and an instruction not to infer from names.
- **Splits:** every source is split by the same hash of the user id, 85% training, 5% validation, 10% test, so no test
  user is trained on through another source. Located users: 962,649 / 56,941 / 113,269 (the most frequent labels are IN
  at 15.4%, US, BR, CN, DE, GB, CA and FR); domain rules: 58,206 / 3,458 / 6,935; teacher: 7,817 / 450 / 916.
- **GeoNames cities:** 23,362 synthetic `location: <city>` rows from the
  [GeoNames](https://www.geonames.org/) cities with at least 15,000 inhabitants (CC BY 4.0), training only, so that
  cities no user wrote are still known. With them the label set covers 245 countries and territories (plus `EU`, from a
  single teacher label). They train the country only, since GeoNames and Nominatim name places differently.

### Training

- **Architecture:** [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
  (Wang et al., 2024; 12 layers, hidden size 384, 118M parameters), mean pooling over the attention mask, dropout 0.1,
  one linear layer over the 246 labels, a second one over 1,148 city classes (the 1,025 cities with at least 50
  training users, plus one "other city" class for each of their 123 countries), a third one over 828 state classes
  (701 states, plus "other state" for each of their 127 countries), and the `query: ` prefix the base model expects.
  The city loss is the cross-entropy over the city classes of the user's labelled country, added to the country loss; a
  location that names no city, or an unknown one, trains its country's "other city" class. The state loss works the
  same way.
- **Fine-tuning:** all encoder layers except the word embeddings, one epoch over the 1,052,034 training examples,
  batch 32, AdamW (weight decay 0.01), learning rates 5e-5 for the encoder and 1e-3 for the head, linear schedule with
  6% warm-up, gradient clipping at 1.0: 32,877 steps, about 12 minutes on one GPU in bf16.
- **Location dropout:** the `location` line is removed from 20% of the located training profiles (170,426), so the model
  also learns from the other fields what a located user's profile says without it.
- **Calibration:** temperature scaling (T = 1.033) fitted on the 56,941 validation users by minimum negative
  log-likelihood, with the same 20% location dropout. The threshold is the lowest value (on a 0.01 grid, 0.001 from
  0.95) whose validation precision reaches 95% both for users who kept their location and for the 10,090 whose location
  was removed: 0.94. Users who keep their location reach 95% at any threshold; the location-hidden ones set it. The
  city and state heads have their own temperatures (T = 1.040 and 1.054, on P(city | labelled country) and
  P(state | labelled country)) and thresholds, chosen the same way on their answers: 0.994 and 0.997.

### Compression

- **Vocabulary trimming.** 96M of the base model's 118M parameters are the embedding table of a 250,037-token
  multilingual vocabulary, most of which GitHub profiles never use. The export keeps only the 74,332 tokens that occur
  at least twice in the inputs of all 5,787,635 profiles. The tokenizer is untouched: the graph starts with a lookup
  from each original token id to its row in the smaller table, and other ids read `<unk>`. On covered texts the trimmed
  network reproduces the full network's logits to within 1.6e-5.
- **Quantization.** ONNX Runtime dynamic quantization (int8 weights, activations quantized at run time) takes the trimmed
  network from 205 MB to 52 MB. Because activation scales are computed per run, the package classifies one profile at a
  time, so an answer never depends on what else is in a batch.

## Limitations and responsible use

- **Estimates, not facts.** The model infers a country from company names, email domains, languages and places in a bio.
  Use it for aggregate questions (where a project's contributors are, how a community is distributed), not to label or
  make decisions about individuals.
- **Calibrated on users who declare a location.** The threshold is set on located users with their location removed,
  and holds on the teacher and domain test sets, but those only cover profiles that name a place somewhere. Most
  location-less profiles name none; for them `null` is the expected answer, and precision on a random sample of them
  is not yet measured.
- **A snapshot of two months of activity.** Labels come from users active in August and September 2026 and from Nominatim's reading
  of their locations. Countries with few users then were learned from few examples, and tokens first seen after the
  collection read as unknown.
- **City names are Nominatim's**, not identifiers: "Munich", "Special Capital Region of Jakarta". Where Nominatim
  gives no city (it does not for "Москва"), the city is learned as unknown, and districts or municipalities can split
  one city into several classes.
- **Names are excluded on purpose.** The login and the name correlate with country through names and ethnicity, which
  this model does not use as evidence.

## API

- **`loadCountryClassifier(source?, options?)`**, from `/node` or `/web`, returns `{ config, classify }`. Without
  `source` it loads the included model. `source` may also be an exported model directory (Node) or its URL (browser):
  `classifier.json`, `tokenizer.json`, `tokenizer_config.json` and `{ onnxFile }`, default `model.onnx` in Node and
  `model_int8.onnx` in the browser. Or it may be a `ModelFiles` object with the path or URL of each file.
- **`classify(texts)`** answers in order: `country_code` (the top country, or `null` below `threshold`; the included
  model's threshold is 0.94, see [Results](#results)), `confidence` and `top` (three most probable countries), then
  `city`, `city_confidence` and `city_top` for the top country (`null`, `null`, `[]` without a city head), and
  `state`, `state_confidence` and `state_top` likewise (see [How it works](#how-it-works)).
- **Options:** `threshold`, `batchSize`; Node: `threads`; browser: `onProgress`, `cache` (Cache API name, or `false`),
  `wasmPaths`, `executionProviders`, `fetch`.
- **The core entry** (`@hsborges-msr/github-country-classifier`) needs no runtime: `describeCountryInput`,
  `createClassifierFromArtifacts`, `createCountryClassifier` (bring your own `RunLogits`), `predictFromLogits`,
  `parseClassifierConfig`.

### In the browser

The `/web` loader downloads the model once (about 69 MB, less with HTTP compression), keeps it in the Cache API and
reports progress through `onProgress`. The model files are referenced with `new URL("…", import.meta.url)`, so your
bundler emits them as assets.

- **Vite:** exclude `onnxruntime-web` from `optimizeDeps`. ONNX Runtime starts its WebAssembly threads as module workers
  of the chunk that contains it, so that chunk must not import your entry chunk; see [`vite.config.ts`](vite.config.ts).
- **Threads** need a cross-origin-isolated page (`Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: credentialless` or `require-corp`); otherwise ONNX Runtime runs single-threaded.

## Demo

The [live demo](https://hsborges-msr.github.io/github-country-classifier/) (source in [`demo/`](demo), not in the
tarball) reads the latest [public GitHub events](https://docs.github.com/rest/activity/events#list-public-events),
fetches each author's profile, classifies it in the browser and maps the result, with a refresh button, a
minimum-confidence slider, event counts per user, and totals for all users and at the chosen confidence.

- Each refresh reads the 300 latest events GitHub keeps (three pages of 100). Without a token GitHub allows 60 requests
  per hour (one per events page, one per profile), so a refresh classifies 20 profiles; with a token (kept in
  `sessionStorage`) it classifies up to 300, every actor of those events.
- Profiles are fetched six at a time through Octokit with throttling and retries; an exhausted quota stops the refresh
  at once instead of waiting up to an hour, and the next refresh picks up where it stopped.
- GitHub Pages cannot send cross-origin isolation headers, so the hosted demo runs on one thread; locally, `yarn demo`
  sends them. `?model=<url>` loads another exported model.

## Development

Clone with [Git LFS](https://git-lfs.com/) installed (the model files are in LFS), then `yarn install`. Development
uses Node 22 (a demo dependency needs it); the package itself supports Node ≥ 20.17.

- `yarn typecheck`, `yarn test` (Vitest, including parity tests against the Python runtime on invented profiles),
  `yarn build`, `yarn demo`, `yarn demo:build`.
- `yarn smoke:pack` packs the package, installs the tarball into a fresh npm project and classifies a profile with both
  entries.
- **Model updates** come from the study repository
  [`hsborges-msr/github-location-inference`](https://github.com/hsborges-msr/github-location-inference) (private; data,
  training code and design records), which includes this repository as a submodule: a new trimmed export is copied into
  `model/` together with updated numbers here and a version bump.
- **Releases:** bump `version` and push the tag `v<version>`; [`release.yml`](.github/workflows/release.yml) tests,
  smoke-tests and attaches the tarball to a GitHub release. [`ci.yml`](.github/workflows/ci.yml) checks every push and
  [`pages.yml`](.github/workflows/pages.yml) publishes the demo.

## Citation

If you use this classifier in research, please cite it (GitHub's "Cite this repository" button reads
[`CITATION.cff`](CITATION.cff)):

```bibtex
@software{borges_github_country_classifier_2026,
  author  = {Borges, Hudson Silva},
  title   = {{GitHub Country Classifier}: Inferring a Developer's Country from Public GitHub Profiles},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/hsborges-msr/github-country-classifier}
}
```

## Acknowledgements and license

MIT, for the code and the model. Built on [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(MIT; L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder and F. Wei, *Multilingual E5 Text Embeddings: A Technical
Report*, arXiv:2402.05672, 2024), [ONNX Runtime](https://onnxruntime.ai/) and
[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers.js). Event data from
[GH Archive](https://www.gharchive.org/); locations geocoded with [Nominatim](https://nominatim.org/), data ©
[OpenStreetMap](https://www.openstreetmap.org/copyright) contributors. City names for training from
[GeoNames](https://www.geonames.org/) (CC BY 4.0).
