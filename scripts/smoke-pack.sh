#!/bin/sh
# Release check: pack this package (code and model), install the tarball and the ONNX runtimes into a new npm project,
# and classify a profile with the Node entry and with the web entry (run in Node). Needs network access to the npm
# registry for the runtimes and the dependencies.
#
# Usage: yarn smoke:pack [--keep]
#   --keep   keep the temporary project and print its path (default: removed)
set -eu

package_dir=$(cd "$(dirname "$0")/.." && pwd)
keep=no
[ "${1:-}" = "--keep" ] && keep=yes

work=$(mktemp -d "${TMPDIR:-/tmp}/country-classifier-smoke.XXXXXX")
cleanup() { if [ "$keep" = yes ]; then echo "kept $work"; else rm -rf "$work"; fi; }
trap cleanup EXIT

(cd "$package_dir" && npm pack --silent --pack-destination "$work" >/dev/null)
cd "$work"
ls -l ./*.tgz
npm init -y >/dev/null
npm install --silent --no-audit --no-fund ./hsborges-msr-github-country-classifier-*.tgz onnxruntime-node@^1.30.0 onnxruntime-web@^1.30.0

cat > check.mjs <<'EOF'
import { readFile } from "node:fs/promises";
import { describeCountryInput, loadCountryClassifier as loadNode } from "@hsborges-msr/github-country-classifier/node";
import { loadCountryClassifier as loadWeb } from "@hsborges-msr/github-country-classifier/web";

const text = describeCountryInput({ location: "Recife, Brasil", company: "CESAR" });
// Node's fetch does not read file: URLs, which is where the bundled model is outside a bundler.
const fetchFile = async url => new Response(await readFile(new URL(String(url))));
for (const [name, load] of [["node", () => loadNode()], ["web", () => loadWeb(undefined, { cache: false, fetch: fetchFile })]]) {
  const [prediction] = await (await load()).classify([text]);
  console.log(`${name}: ${JSON.stringify(prediction)}`);
  if (prediction?.country_code !== "BR") throw new Error(`${name} entry answered ${prediction?.country_code}`);
}
EOF
node check.mjs
