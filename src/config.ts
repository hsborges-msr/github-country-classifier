import { z } from "zod";
import { COUNTRY_INPUT_FIELDS } from "./input.js";

const countryCode = z.string().regex(/^[A-Z]{2}$/u);

const baseConfigSchema = z.object({
  base_model: z.string().min(1),
  prefix: z.string(),
  max_length: z.number().int().min(3),
  pooling: z.literal("mean"),
  input_fields: z.array(z.string()),
  labels: z.array(countryCode).min(1),
  temperature: z.number().positive(),
  threshold: z.number().min(0).max(1),
  target_precision: z.number().min(0).max(1),
});

/** City or state classes as `[country, name]`; a null name is the country's "other" class. */
const subClasses = z.array(z.tuple([countryCode, z.string().min(1).nullable()])).min(1);
const cityHead = { cities: subClasses, city_temperature: z.number().positive(), city_threshold: z.number().min(0).max(1) };
const noCityHead = { cities: z.never().optional() };
const stateHead = { states: subClasses, state_temperature: z.number().positive(), state_threshold: z.number().min(0).max(1) };
const noStateHead = { states: z.never().optional() };

/**
 * `classifier.json` of a trained model (README, "How it works"); unknown keys are ignored. Format 2 adds a city head,
 * format 3 a state head and possibly a city head.
 */
const classifierConfigSchema = z
  .union([
    baseConfigSchema.extend({ format: z.literal(1), ...noCityHead, ...noStateHead }),
    baseConfigSchema.extend({ format: z.literal(2), ...cityHead, ...noStateHead }),
    baseConfigSchema.extend({ format: z.literal(3), ...cityHead, ...stateHead }),
    baseConfigSchema.extend({ format: z.literal(3), ...noCityHead, ...stateHead }),
  ])
  .refine(config => (config.cities ?? []).every(([code]) => config.labels.includes(code)), { message: "cities must belong to a label", path: ["cities"] })
  .refine(config => (config.states ?? []).every(([code]) => config.labels.includes(code)), { message: "states must belong to a label", path: ["states"] });

export type ClassifierConfig = z.infer<typeof classifierConfigSchema>;

/** Validate a classifier config; a model trained on other input fields would read different text, so it is rejected. */
export function parseClassifierConfig(value: unknown): ClassifierConfig {
  const config = classifierConfigSchema.parse(value);
  if (config.input_fields.join(",") !== COUNTRY_INPUT_FIELDS.join(",")) {
    throw new Error(`classifier input fields ${config.input_fields.join(",")} differ from ${COUNTRY_INPUT_FIELDS.join(",")}: export a new dataset and retrain`);
  }
  return config;
}
