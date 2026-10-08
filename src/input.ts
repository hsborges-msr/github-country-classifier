/**
 * The profile fields the country classifier reads, in input order (ADR 0007). `email_domain` is derived from the public
 * email; `login` and `name` are never read. Changing the fields or their order changes the model contract: export a new
 * dataset and retrain.
 */
export const COUNTRY_INPUT_FIELDS = ["location", "company", "blog", "email_domain", "twitter_username", "bio"] as const;

/** The public GitHub profile fields the classifier input is built from; missing and null fields are skipped. */
export type CountryProfileFields = {
  location?: string | null | undefined;
  company?: string | null | undefined;
  blog?: string | null | undefined;
  email?: string | null | undefined;
  twitter_username?: string | null | undefined;
  bio?: string | null | undefined;
};

const HOSTNAME = /^[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+$/u;

/** The lowercase domain after the last `@` of an email, without a trailing dot; null when empty or not a dotted hostname. */
export function emailDomain(email: string | null | undefined): string | null {
  if (email === null || email === undefined) return null;
  const at = email.lastIndexOf("@");
  if (at < 0) return null;
  const host = email.slice(at + 1).trim().toLowerCase().replace(/\.+$/u, "");
  return HOSTNAME.test(host) ? host : null;
}

/**
 * The classifier's input text: one `field: value` line per non-empty field of {@link COUNTRY_INPUT_FIELDS}, with
 * whitespace runs collapsed to one space and trimmed. Empty when no field has text, in which case the answer is null
 * without running the model. The same function builds the training data and the inference input.
 */
export function describeCountryInput(profile: CountryProfileFields): string {
  return describeFields(COUNTRY_INPUT_FIELDS, profile);
}

/** One `field: value` line per non-empty string field, in order, whitespace collapsed; `email_domain` is the domain of `email`. */
export function describeFields(fields: readonly string[], profile: Readonly<Record<string, unknown>>): string {
  return fields.flatMap(field => {
    const raw = field === "email_domain" ? emailDomain(typeof profile.email === "string" ? profile.email : null) : profile[field];
    const value = typeof raw === "string" ? raw.replace(/\s+/gu, " ").trim() : "";
    return value ? [`${field}: ${value}`] : [];
  }).join("\n");
}
