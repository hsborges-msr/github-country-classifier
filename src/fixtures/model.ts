import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

/**
 * The full fp32 export of the training run, where this repository is checked out as `packages/country-classifier` of
 * the study repository; the fp32 runtime tests are skipped when it is absent (as in a standalone checkout).
 */
export const MODEL_DIR = fileURLToPath(new URL("../../../../training/outputs/e5-small-v2/onnx/", import.meta.url));
export const HAS_MODEL = existsSync(`${MODEL_DIR}model.onnx`);

/**
 * Invented profile texts with the answers of the fp32 model run by the Python training code (ONNX Runtime): confident
 * cases across countries and scripts, four without a place signal (low confidence) and one longer than the 96 tokens.
 */
export interface ParityCase {
  text: string;
  country_code: string;
  confidence: number;
}

export async function readParityCases(): Promise<ParityCase[]> {
  return JSON.parse(await readFile(new URL("./parity.json", import.meta.url), "utf8")) as ParityCase[];
}
