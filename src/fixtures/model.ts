import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

/** The exported model of the repository's training run; the runtime tests are skipped when it is absent. */
export const MODEL_DIR = fileURLToPath(new URL("../../../../training/outputs/e5-small/onnx/", import.meta.url));
export const HAS_MODEL = existsSync(`${MODEL_DIR}model.onnx`);

/** Test texts with the predictions of the Python evaluation (`location_ft.evaluate --onnx model.onnx`). */
export interface ParityCase {
  text: string;
  country_code: string;
  confidence: number;
}

export async function readParityCases(): Promise<ParityCase[]> {
  return JSON.parse(await readFile(new URL("./parity.json", import.meta.url), "utf8")) as ParityCase[];
}
