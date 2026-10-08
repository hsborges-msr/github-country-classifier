import { Octokit } from "@octokit/core";
import { retry } from "@octokit/plugin-retry";
import { throttling } from "@octokit/plugin-throttling";
import { RequestError } from "@octokit/request-error";
import { z } from "zod";

const GitHub = Octokit.plugin(throttling, retry);
const API_VERSION = "2026-03-10";
/** Longest secondary-limit pause the page waits out (seconds); primary limits last up to an hour and are never waited out. */
const MAX_SECONDARY_WAIT = 60;

const eventSchema = z.object({
  id: z.string(),
  type: z.string().nullish(),
  actor: z.object({ login: z.string(), avatar_url: z.string() }),
});

const userSchema = z.object({
  login: z.string(),
  html_url: z.string(),
  avatar_url: z.string(),
  location: z.string().nullish(),
  company: z.string().nullish(),
  blog: z.string().nullish(),
  email: z.string().nullish(),
  twitter_username: z.string().nullish(),
  bio: z.string().nullish(),
});

export type GitHubEvent = z.infer<typeof eventSchema>;
export type GitHubUser = z.infer<typeof userSchema>;

export interface RateLimit {
  remaining: number;
  limit: number;
  resetAt: Date;
}

export class RateLimitError extends Error {
  constructor(readonly resetAt: Date | null) {
    super(`GitHub API rate limit exhausted${resetAt ? ` until ${resetAt.toLocaleTimeString()}` : ""}`);
  }
}

type Headers = Record<string, string | number | undefined>;

function readRateLimit(headers: Headers): RateLimit | null {
  const [remaining, limit, reset] = ["x-ratelimit-remaining", "x-ratelimit-limit", "x-ratelimit-reset"].map(name => Number(headers[name]));
  if (headers["x-ratelimit-remaining"] === undefined || [remaining, limit, reset].some(value => value === undefined || Number.isNaN(value))) return null;
  return { remaining: remaining as number, limit: limit as number, resetAt: new Date((reset as number) * 1000) };
}

/** A 429, or a 403 with an exhausted quota, a `retry-after` header or a rate-limit message. */
function asRateLimitError(error: unknown): RateLimitError | null {
  if (!(error instanceof RequestError) || (error.status !== 403 && error.status !== 429)) return null;
  const headers: Headers = error.response?.headers ?? {};
  const rateLimited = error.status === 429 || headers["x-ratelimit-remaining"] === "0" || headers["retry-after"] !== undefined || /rate limit/iu.test(error.message);
  return rateLimited ? new RateLimitError(readRateLimit(headers)?.resetAt ?? null) : null;
}

/**
 * The two REST endpoints the demo uses, from the browser, through Octokit with throttling and retries, so callers can
 * run requests in parallel: transient failures are retried, a short secondary limit is waited out once, and an exhausted
 * quota fails at once with {@link RateLimitError}. Without a token GitHub allows 60 requests per hour and never returns
 * emails; `onRateLimit` reports the quota after every response.
 */
export function createGitHubClient(options: {
  token?: string | undefined;
  onRateLimit?: ((rate: RateLimit) => void) | undefined;
  /** Replaces the global `fetch`, for tests. */
  fetch?: typeof fetch | undefined;
  /** Milliseconds multiplied by the squared retry count between retries of transient failures (Octokit's default: 1000). */
  retryAfterBaseValue?: number | undefined;
}) {
  const octokit = new GitHub({
    ...(options.token ? { auth: options.token } : {}),
    request: { cache: "no-store", ...(options.fetch ? { fetch: options.fetch } : {}) },
    log: { debug: () => {}, info: () => {}, warn: console.warn, error: () => {} },
    retry: {
      retries: 2,
      doNotRetry: [400, 401, 403, 404, 410, 422, 429, 451],
      ...(options.retryAfterBaseValue === undefined ? {} : { retryAfterBaseValue: options.retryAfterBaseValue }),
    },
    throttle: {
      onRateLimit: () => false,
      onSecondaryRateLimit: (retryAfter, _request, _octokit, retryCount) => retryCount === 0 && retryAfter <= MAX_SECONDARY_WAIT,
    },
  });
  const report = (headers: Headers | undefined) => {
    const rate = headers && readRateLimit(headers);
    if (rate) options.onRateLimit?.(rate);
  };
  octokit.hook.before("request", request => {
    request.headers["x-github-api-version"] = API_VERSION;
  });
  octokit.hook.after("request", response => report(response.headers));
  // Only reports the quota: errors must reach the throttling and retry plugins unchanged.
  octokit.hook.error("request", error => {
    if (error instanceof RequestError) report(error.response?.headers);
    throw error;
  });

  return {
    /** The latest public events (GET /events), newest first. */
    async listPublicEvents(perPage = 100): Promise<GitHubEvent[]> {
      try {
        const response = await octokit.request("GET /events", { per_page: perPage });
        return z.array(eventSchema).parse(response.data);
      } catch (error) {
        throw asRateLimitError(error) ?? error;
      }
    },
    /** A public profile (GET /users/{username}), or null when it no longer exists. */
    async getUser(login: string, signal?: AbortSignal): Promise<GitHubUser | null> {
      try {
        const response = await octokit.request("GET /users/{username}", { username: login, request: signal ? { signal } : {} });
        return userSchema.parse(response.data);
      } catch (error) {
        if (error instanceof RequestError && error.status === 404) return null;
        throw asRateLimitError(error) ?? error;
      }
    },
  };
}
