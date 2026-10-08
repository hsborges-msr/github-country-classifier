# Repository notes

- One package, `@hsborges-msr/github-country-classifier`: Node >=20.17.0, ESM with TypeScript `NodeNext` (use `.js`
  extensions in relative imports), Yarn 1. Entries: `src/index.ts` (runtime-free contract), `src/node.ts`
  (`onnxruntime-node`), `src/web.ts` (`onnxruntime-web`); `src/model.ts` locates the included model in `model/`.
- `model/` is produced by the study repository `hsborges-msr/github-location-inference` (training code, data and ADRs),
  which includes this repository as the submodule `packages/country-classifier`. Its large files are in Git LFS. Change
  it only with a new trimmed export from there, the README numbers and a version bump. The input contract
  (`describeCountryInput`, field order, prefix, max length) must match the training data: changing it means retraining.
- `yarn typecheck`, `yarn test` (Vitest over `src/` and `demo/`), `yarn build`, `yarn demo`, `yarn smoke:pack` (packs
  and installs the tarball into a new npm project). Tests sit beside their source as `*.spec.ts`, are deterministic and
  never call live services; write a failing test first for behavior changes.
- Validate external input with Zod and infer types from schemas; keep I/O at the edges and prefer small pure functions.
- Distribution is a tarball on GitHub Releases (tag `v<version>`, `.github/workflows/release.yml`), not npm.
- `src/fixtures/parity.json` holds invented profiles only; never add real profile text to this public repository.
