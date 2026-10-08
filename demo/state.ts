import type { CountryPrediction } from "../src/index.js";

/** The fields of a public event the demo reads. */
export interface EventLike {
  id: string;
  type?: string | null | undefined;
  actor: { login: string; avatar_url: string };
}

export interface Actor {
  login: string;
  avatarUrl: string;
}

/** Events seen per actor and event type (`PushEvent`, …), and the ids already counted. */
export interface ActivityState {
  activity: ReadonlyMap<string, ReadonlyMap<string, number>>;
  seenEventIds: ReadonlySet<string>;
}

/** Profiles per refresh: anonymous requests are limited to 60 per hour; with a token (5,000 per hour) a whole events page. */
export const PROFILE_LIMIT = { anonymous: 20, token: 100 } as const;

/** The profile limit after a token is entered or removed; a number the user picked (not a default) is kept. */
export function nextProfileLimit(current: number, hasToken: boolean): number {
  const isDefault = current === PROFILE_LIMIT.anonymous || current === PROFILE_LIMIT.token;
  if (!isDefault) return current;
  return hasToken ? PROFILE_LIMIT.token : PROFILE_LIMIT.anonymous;
}

const isBot = (login: string) => login.endsWith("[bot]");

/**
 * Count the events not counted yet (consecutive pages of `GET /events` overlap) per human actor and type. Returns a new
 * state; the given one is not changed.
 */
export function addActivity(state: ActivityState, events: readonly EventLike[]): ActivityState {
  const activity = new Map([...state.activity].map(([login, types]) => [login, new Map(types)]));
  const seenEventIds = new Set(state.seenEventIds);
  for (const { id, type, actor } of events) {
    if (seenEventIds.has(id) || isBot(actor.login)) continue;
    seenEventIds.add(id);
    const types = activity.get(actor.login) ?? new Map<string, number>();
    const key = type ?? "Event";
    types.set(key, (types.get(key) ?? 0) + 1);
    activity.set(actor.login, types);
  }
  return { activity, seenEventIds };
}

/** Total events and `[type without the "Event" suffix, count]`, most frequent first, ties by name. */
export function describeActivity(types: ReadonlyMap<string, number> | undefined): { total: number; types: Array<[string, number]> } {
  const entries = [...(types ?? [])].map(([type, count]): [string, number] => [type.replace(/Event$/u, "") || type, count]);
  entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return { total: entries.reduce((sum, [, count]) => sum + count, 0), types: entries };
}

/** A classified event actor. */
export interface ActorResult extends Actor {
  htmlUrl: string;
  /** The classifier input built from the profile (empty when the profile has none of its fields). */
  text: string;
  prediction: CountryPrediction;
}

/** Each actor not in `known`, in the order of their newest event as GitHub lists them, without bots, at most `limit`. */
export function newActors(events: readonly EventLike[], known: ReadonlySet<string>, limit: number): Actor[] {
  const seen = new Set(known);
  const actors: Actor[] = [];
  for (const { actor } of events) {
    if (actors.length >= limit) break;
    if (seen.has(actor.login) || isBot(actor.login)) continue;
    seen.add(actor.login);
    actors.push({ login: actor.login, avatarUrl: actor.avatar_url });
  }
  return actors;
}

/** The most probable country when its calibrated probability reaches `minConfidence`, else null (indeterminate). */
export function countryOf(result: ActorResult, minConfidence: number): string | null {
  const [best] = result.prediction.top;
  return best !== undefined && best[1] >= minConfidence ? best[0] : null;
}

export interface Summary {
  /** Classified users, and the events seen from them. */
  users: number;
  events: number;
  /** Users answered with a country at the minimum confidence, and their events. */
  placed: number;
  placedEvents: number;
  countries: number;
  /** Users below the minimum confidence or with an empty profile. */
  indeterminate: number;
}

/** Totals of the classified users at `minConfidence`; at 0 every non-empty profile is placed. */
export function summarize(results: readonly ActorResult[], activity: ActivityState["activity"], minConfidence: number): Summary {
  const eventsOf = (login: string) => describeActivity(activity.get(login)).total;
  const placed = results.filter(result => countryOf(result, minConfidence) !== null);
  return {
    users: results.length,
    events: results.reduce((sum, result) => sum + eventsOf(result.login), 0),
    placed: placed.length,
    placedEvents: placed.reduce((sum, result) => sum + eventsOf(result.login), 0),
    countries: tallyCountries(results, minConfidence).size,
    indeterminate: results.length - placed.length,
  };
}

/** Actors per answered country, most frequent first, ties by country code. */
export function tallyCountries(results: readonly ActorResult[], minConfidence: number): Map<string, number> {
  const counts = new Map<string, number>();
  for (const result of results) {
    const code = countryOf(result, minConfidence);
    if (code !== null) counts.set(code, (counts.get(code) ?? 0) + 1);
  }
  return new Map([...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])));
}
