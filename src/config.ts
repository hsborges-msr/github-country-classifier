import { z } from "zod";
import { COUNTRY_INPUT_FIELDS } from "./input.js";

/** `classifier.json` of a trained model (README, "How it works"); unknown keys are ignored. */
const classifierConfigSchema = z.object({
  format: z.literal(1),
  base_model: z.string().min(1),
  prefix: z.string(),
  max_length: z.number().int().min(3),
  pooling: z.literal("mean"),
  input_fields: z.array(z.string()),
  labels: z.array(z.string().regex(/^[A-Z]{2}$/u)).min(1),
  temperature: z.number().positive(),
  threshold: z.number().min(0).max(1),
  target_precision: z.number().min(0).max(1),
});

export type ClassifierConfig = z.infer<typeof classifierConfigSchema>;

/** Validate a classifier config; a model trained on other input fields would read different text, so it is rejected. */
export function parseClassifierConfig(value: unknown): ClassifierConfig {
  const config = classifierConfigSchema.parse(value);
  if (config.input_fields.join(",") !== COUNTRY_INPUT_FIELDS.join(",")) {
    throw new Error(`classifier input fields ${config.input_fields.join(",")} differ from ${COUNTRY_INPUT_FIELDS.join(",")}: export a new dataset and retrain`);
  }
  return config;
}
