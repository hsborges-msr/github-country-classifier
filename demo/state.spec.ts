import { expect, test } from "vitest";
import { addActivity, countryOf, describeActivity, newActors, nextEventLimit, placeOf, summarize, tallyCountries, type ActorResult } from "./state.js";

const event = (login: string, type: string | null = "PushEvent", id = `${login}-${type}`) => ({ id, type, actor: { login, avatar_url: `https://a/${login}` } });

test("newActors keeps each unseen human actor once, in order", () => {
  const events = [event("ana"), event("dependabot[bot]"), event("bo", "IssuesEvent"), event("ana", "WatchEvent"), event("cy"), event("di")];
  expect(newActors(events, new Set(["cy"]))).toEqual([
    { login: "ana", avatarUrl: "https://a/ana" },
    { login: "bo", avatarUrl: "https://a/bo" },
    { login: "di", avatarUrl: "https://a/di" },
  ]);
});

test("nextEventLimit follows the token unless the user picked another number", () => {
  expect(nextEventLimit(30, true)).toBe(300);
  expect(nextEventLimit(300, false)).toBe(30);
  expect(nextEventLimit(300, true)).toBe(300);
  expect(nextEventLimit(35, true)).toBe(35);
  expect(nextEventLimit(35, false)).toBe(35);
});

test("addActivity counts each new event once per actor and type, skipping bots, without changing its inputs", () => {
  const first = addActivity({ activity: new Map(), seenEventIds: new Set() }, [event("ana", "PushEvent", "1"), event("ana", "PushEvent", "2"), event("bo", "IssuesEvent", "3"), event("x[bot]", "PushEvent", "4")]);
  expect(first.activity).toEqual(new Map([["ana", new Map([["PushEvent", 2]])], ["bo", new Map([["IssuesEvent", 1]])]]));
  // The next page overlaps the previous one: events 2 and 3 are not counted again.
  const second = addActivity(first, [event("ana", "PullRequestEvent", "5"), event("ana", "PushEvent", "2"), event("bo", "IssuesEvent", "3"), event("cy", null, "6")]);
  expect(second.activity).toEqual(
    new Map([
      ["ana", new Map([["PushEvent", 2], ["PullRequestEvent", 1]])],
      ["bo", new Map([["IssuesEvent", 1]])],
      ["cy", new Map([["Event", 1]])],
    ]),
  );
  expect(second.seenEventIds).toEqual(new Set(["1", "2", "3", "5", "6"]));
  expect(first.activity.get("ana")).toEqual(new Map([["PushEvent", 2]]));
  expect(first.seenEventIds.size).toBe(3);
});

test("describeActivity totals the events and lists the types, most frequent first", () => {
  expect(describeActivity(new Map([["PushEvent", 1], ["PullRequestEvent", 3], ["IssuesEvent", 1]]))).toEqual({
    total: 5,
    types: [["PullRequest", 3], ["Issues", 1], ["Push", 1]],
  });
  expect(describeActivity(undefined)).toEqual({ total: 0, types: [] });
});

const result = (login: string, top: Array<[string, number]>): ActorResult => ({
  login,
  avatarUrl: "",
  htmlUrl: "",
  text: top.length ? "bio: x" : "",
  prediction: { country_code: top[0]?.[0] ?? null, confidence: top[0]?.[1] ?? null, top, city: null, city_confidence: null, city_top: [], state: null, state_confidence: null, state_top: [] },
});

test("countryOf answers the top label only when it reaches the minimum confidence", () => {
  expect(countryOf(result("a", [["BR", 0.8]]), 0.5)).toBe("BR");
  expect(countryOf(result("a", [["BR", 0.4]]), 0.5)).toBeNull();
  expect(countryOf(result("a", []), 0)).toBeNull();
});

test("placeOf names the answered city and state of a placed country, once when they are the same", () => {
  const placed = (city: string | null, state: string | null) => {
    const base = result("a", [["BR", 0.99]]);
    return { ...base, prediction: { ...base.prediction, city, state } };
  };
  expect(placeOf(placed("Recife", "Pernambuco"), 0.5)).toBe("Recife, Pernambuco");
  expect(placeOf(placed("Tokyo", "Tokyo"), 0.5)).toBe("Tokyo");
  expect(placeOf(placed(null, "Pernambuco"), 0.5)).toBe("Pernambuco");
  expect(placeOf(placed("Recife", null), 0.5)).toBe("Recife");
  expect(placeOf(placed(null, null), 0.5)).toBeNull();
  // Below the minimum confidence the country is indeterminate, so its city and state are not shown either.
  expect(placeOf(placed("Recife", "Pernambuco"), 0.995)).toBeNull();
});

test("summarize counts users, placed users, countries and their events at a minimum confidence", () => {
  const results = [result("a", [["US", 0.9]]), result("b", [["BR", 0.4]]), result("c", [["US", 0.7]]), result("d", [])];
  const activity = new Map([
    ["a", new Map([["PushEvent", 3]])],
    ["b", new Map([["PushEvent", 1], ["IssuesEvent", 1]])],
    ["c", new Map([["WatchEvent", 1]])],
    ["d", new Map([["PushEvent", 4]])],
    ["not-classified", new Map([["PushEvent", 9]])],
  ]);
  // At 0 every non-empty profile is placed: only the empty one is indeterminate.
  expect(summarize(results, activity, 0)).toEqual({ users: 4, events: 10, placed: 3, placedEvents: 6, countries: 2, indeterminate: 1 });
  expect(summarize(results, activity, 0.5)).toEqual({ users: 4, events: 10, placed: 2, placedEvents: 4, countries: 1, indeterminate: 2 });
  expect(summarize([], activity, 0.5)).toEqual({ users: 0, events: 0, placed: 0, placedEvents: 0, countries: 0, indeterminate: 0 });
});

test("tallyCountries counts answered actors, most frequent first, ties by code", () => {
  const results = [result("a", [["US", 0.9]]), result("b", [["BR", 0.9]]), result("c", [["US", 0.7]]), result("d", [["DE", 0.2]]), result("e", []), result("f", [["AR", 0.9]])];
  expect([...tallyCountries(results, 0.5)]).toEqual([["US", 2], ["AR", 1], ["BR", 1]]);
});
