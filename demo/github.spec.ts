import { expect, test, vi } from "vitest";
import { createGitHubClient, RateLimitError } from "./github.js";

const RESET = 1_800_000_000;
const quota = (remaining: number) => ({ "x-ratelimit-remaining": String(remaining), "x-ratelimit-limit": "60", "x-ratelimit-reset": String(RESET) });
const json = (body: unknown, status = 200, headers: Record<string, string> = quota(59)) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
const user = { login: "ana", html_url: "https://github.com/ana", avatar_url: "https://a/ana", location: "Recife", extra: 1 };

/** Answers each request with the next response of `responses`. */
function fakeFetch(...responses: Response[]) {
  return vi.fn(async (_url: string | URL | Request) => responses.shift() ?? new Response(null, { status: 599 }));
}

test("getUser returns the projected profile, sends the API version and reports the quota", async () => {
  const fetch = fakeFetch(json(user));
  const onRateLimit = vi.fn();
  const github = createGitHubClient({ fetch, onRateLimit, token: "t0ken" });
  expect(await github.getUser("ana")).toEqual({ login: "ana", html_url: "https://github.com/ana", avatar_url: "https://a/ana", location: "Recife" });
  const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe("https://api.github.com/users/ana");
  expect(init.headers).toMatchObject({ "x-github-api-version": "2026-03-10", authorization: "token t0ken" });
  expect(onRateLimit).toHaveBeenCalledWith({ remaining: 59, limit: 60, resetAt: new Date(RESET * 1000) });
});

test("getUser answers null for a deleted profile", async () => {
  expect(await createGitHubClient({ fetch: fakeFetch(json({ message: "Not Found" }, 404)) }).getUser("gone")).toBeNull();
});

test("an exhausted quota fails at once with RateLimitError instead of waiting for the reset", async () => {
  const fetch = fakeFetch(json({ message: "API rate limit exceeded" }, 403, quota(0)));
  const onRateLimit = vi.fn();
  const error = await createGitHubClient({ fetch, onRateLimit }).getUser("ana").catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(RateLimitError);
  expect((error as RateLimitError).resetAt).toEqual(new Date(RESET * 1000));
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(onRateLimit).toHaveBeenCalledWith(expect.objectContaining({ remaining: 0 }));
});

test("transient failures are retried; other errors are thrown", async () => {
  const fetch = fakeFetch(json({ message: "boom" }, 502), json([{ id: "1", type: "PushEvent", actor: { login: "ana", avatar_url: "https://a/ana" } }]));
  const github = createGitHubClient({ fetch, retryAfterBaseValue: 1 });
  expect(await github.listPublicEvents()).toEqual([{ id: "1", type: "PushEvent", actor: { login: "ana", avatar_url: "https://a/ana" } }]);
  expect(fetch).toHaveBeenCalledTimes(2);
  await expect(createGitHubClient({ fetch: fakeFetch(json({ message: "Bad credentials" }, 401)) }).getUser("ana")).rejects.toMatchObject({ status: 401 });
});

test("an aborted signal cancels the request", async () => {
  const controller = new AbortController();
  controller.abort(new Error("stop"));
  const fetch = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
    init?.signal?.throwIfAborted();
    return json(user);
  });
  await expect(createGitHubClient({ fetch }).getUser("ana", controller.signal)).rejects.toThrow();
});
