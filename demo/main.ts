import "./style.css";
import { describeCountryInput, loadCountryClassifier, type LoadProgress } from "../src/web.js";
import { forEachConcurrently } from "./concurrency.js";
import { createGitHubClient, RateLimitError, type RateLimit } from "./github.js";
import { createWorldMap, type WorldMap } from "./map.js";
import { addActivity, countryOf, describeActivity, newActors, nextProfileLimit, PROFILE_LIMIT, summarize, tallyCountries, type ActivityState, type ActorResult } from "./state.js";

// The bundled model by default; `?model=<url>` points at another exported model directory, `?onnx=model.onnx` picks its fp32 file.
const params = new URLSearchParams(location.search);
const MODEL_URL = params.get("model") ?? undefined;
const ONNX_FILE = params.get("onnx") ?? "model_int8.onnx";
const MODEL_NAME = MODEL_URL === undefined ? "bundled int8 model" : `${MODEL_URL} ${ONNX_FILE}`;
const TOKEN_KEY = "github-country-classifier:token";
const MAX_LISTED_USERS = 200;
/** Profile lookups in flight; Octokit's throttling also caps concurrent requests and paces secondary limits. */
const PARALLEL_REQUESTS = 6;

function byId<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`#${id} is missing`);
  return element as unknown as T;
}

/** Create an element; strings become text nodes, so profile text is never parsed as HTML. */
function h<K extends keyof HTMLElementTagNameMap>(tag: K, attributes: Record<string, string> = {}, ...children: Array<Node | string>): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  element.append(...children);
  return element;
}

const ui = {
  toolbar: byId<HTMLFormElement>("toolbar"),
  refresh: byId<HTMLButtonElement>("refresh"),
  clear: byId<HTMLButtonElement>("clear"),
  limit: byId<HTMLInputElement>("limit"),
  confidence: byId<HTMLInputElement>("confidence"),
  confidenceValue: byId<HTMLOutputElement>("confidence-value"),
  token: byId<HTMLInputElement>("token"),
  modelStatus: byId<HTMLSpanElement>("model-status"),
  modelProgress: byId<HTMLProgressElement>("model-progress"),
  apiStatus: byId<HTMLSpanElement>("api-status"),
  runStatus: byId<HTMLSpanElement>("run-status"),
  map: byId<SVGSVGElement>("map"),
  summary: byId<HTMLParagraphElement>("summary"),
  statsAll: byId<HTMLDListElement>("stats-all"),
  statsThreshold: byId<HTMLDListElement>("stats-threshold"),
  statsThresholdTitle: byId<HTMLHeadingElement>("stats-threshold-title"),
  ranking: byId<HTMLOListElement>("ranking"),
  usersTitle: byId<HTMLHeadingElement>("users-title"),
  users: byId<HTMLUListElement>("users"),
};

const regionNames = new Intl.DisplayNames(["en"], { type: "region" });
const countryName = (code: string) => regionNames.of(code) ?? code;
const flag = (code: string) => String.fromCodePoint(...[...code].map(char => 0x1f1e6 + char.charCodeAt(0) - 65));
const percent = (value: number) => `${Math.round(value * 100)}%`;
const plural = (count: number, one: string, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

/** Classified users, newest first. */
const results: ActorResult[] = [];
/** Logins already looked up, including deleted profiles, so a refresh only spends API calls on new actors. */
const known = new Set<string>();
/** Events seen per actor across refreshes, each event counted once. */
let activityState: ActivityState = { activity: new Map(), seenEventIds: new Set() };
let selected: string | null = null;
let worldMap: WorldMap | null = null;
let running = false;

// Model download: the browser keeps the files in the Cache API, so later visits start immediately.
const downloads = new Map<string, LoadProgress>();
function showProgress(progress: LoadProgress) {
  downloads.set(progress.file, progress);
  const loaded = [...downloads.values()].reduce((sum, item) => sum + item.loaded, 0);
  const total = [...downloads.values()].reduce((sum, item) => sum + (item.total ?? item.loaded), 0);
  ui.modelProgress.value = total ? loaded / total : 0;
  ui.modelStatus.textContent = `Loading model… ${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB`;
}

const classifierPromise = loadCountryClassifier(MODEL_URL, { onnxFile: ONNX_FILE, onProgress: showProgress }).then(
  classifier => {
    ui.modelProgress.hidden = true;
    ui.modelStatus.textContent = `Model ready · ${classifier.config.labels.length} countries · ${MODEL_NAME}`;
    return classifier;
  },
  (error: unknown) => {
    ui.modelProgress.hidden = true;
    ui.modelStatus.textContent = `Model failed to load (${MODEL_NAME}): ${error instanceof Error ? error.message : String(error)}`;
    ui.modelStatus.classList.add("error");
    throw error;
  },
);

function showRateLimit(rate: RateLimit) {
  ui.apiStatus.textContent = `GitHub API: ${rate.remaining}/${rate.limit} left, resets ${rate.resetAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  ui.apiStatus.classList.toggle("error", rate.remaining === 0);
}

function minConfidence(): number {
  return Number(ui.confidence.value);
}

function select(code: string | null) {
  selected = code;
  render();
}

function renderRanking(counts: ReadonlyMap<string, number>) {
  const max = Math.max(1, ...counts.values());
  ui.ranking.replaceChildren(
    ...[...counts].map(([code, count]) => {
      const bar = h("span", { class: "bar" });
      bar.style.width = `${(100 * count) / max}%`;
      const item = h("li", { class: code === selected ? "selected" : "", title: "Show these users" }, h("span", { class: "flag" }, flag(code)), h("span", { class: "name" }, countryName(code)), h("span", { class: "count" }, String(count)), bar);
      item.addEventListener("click", () => select(selected === code ? null : code));
      return item;
    }),
  );
  if (counts.size === 0) ui.ranking.append(h("li", { class: "empty" }, "No country above the minimum confidence yet."));
}

function userCard(result: ActorResult): HTMLLIElement {
  const code = countryOf(result, minConfidence());
  const confidence = result.prediction.confidence;
  const answer = code
    ? h("span", { class: "answer" }, `${flag(code)} ${countryName(code)}`, h("small", {}, ` ${percent(confidence ?? 0)}`))
    : h("span", { class: "answer indeterminate" }, result.text ? `Indeterminate (best: ${result.prediction.top[0]?.[0] ?? "?"} ${percent(confidence ?? 0)})` : "Indeterminate (empty profile)");
  const activity = describeActivity(activityState.activity.get(result.login));
  const activityLine = h("p", { class: "activity" }, activity.types.map(([type, count]) => `${type} ×${count}`).join(" · "));
  const details = h("details", {}, h("summary", {}, "input"), h("pre", {}, result.text || "(no location, company, blog, email, Twitter or bio)"));
  if (result.prediction.top.length > 1) details.append(h("p", { class: "top" }, `Top 3: ${result.prediction.top.map(([label, p]) => `${label} ${percent(p)}`).join(" · ")}`));
  return h(
    "li",
    {},
    h("img", { src: `${result.avatarUrl}${result.avatarUrl.includes("?") ? "&" : "?"}s=80`, alt: "", loading: "lazy", width: "40", height: "40" }),
    h(
      "div",
      { class: "who" },
      h("a", { href: result.htmlUrl, target: "_blank", rel: "noopener" }, result.login),
      h("small", { class: "events", title: "Public events seen by this page since it was opened" }, plural(activity.total, "event")),
    ),
    answer,
    activityLine,
    details,
  );
}

function stat(label: string, value: number): HTMLDivElement {
  return h("div", {}, h("dt", {}, label), h("dd", {}, value.toLocaleString("en")));
}

function render() {
  const threshold = minConfidence();
  ui.confidenceValue.textContent = percent(threshold);
  const counts = tallyCountries(results, threshold);
  worldMap?.update(counts, selected);
  renderRanking(counts);
  const all = summarize(results, activityState.activity, 0);
  const atThreshold = summarize(results, activityState.activity, threshold);
  ui.statsAll.replaceChildren(
    stat("Users", all.users),
    stat("Events", all.events),
    stat("With a country guess", all.placed),
    stat("Countries", all.countries),
    stat("Empty profiles", all.indeterminate),
  );
  ui.statsThresholdTitle.textContent = `At ≥ ${percent(threshold)} confidence`;
  ui.statsThreshold.replaceChildren(
    stat("Placed users", atThreshold.placed),
    stat("Their events", atThreshold.placedEvents),
    stat("Countries", atThreshold.countries),
    stat("Indeterminate", atThreshold.indeterminate),
  );
  ui.summary.textContent = results.length ? "" : "No users yet: press Refresh.";
  const listed = selected === null ? results : results.filter(result => countryOf(result, threshold) === selected);
  ui.usersTitle.textContent = selected === null ? "Classified users" : `Classified users in ${flag(selected)} ${countryName(selected)} (click the country again to show all)`;
  ui.users.replaceChildren(...listed.slice(0, MAX_LISTED_USERS).map(userCard));
}

async function refresh() {
  if (running) return;
  running = true;
  ui.refresh.disabled = true;
  const token = ui.token.value.trim();
  if (token) sessionStorage.setItem(TOKEN_KEY, token);
  else sessionStorage.removeItem(TOKEN_KEY);
  const github = createGitHubClient({ token, onRateLimit: showRateLimit });
  const limit = Math.min(PROFILE_LIMIT.token, Math.max(1, Number(ui.limit.value) || PROFILE_LIMIT.anonymous));
  let added = 0;
  try {
    ui.runStatus.textContent = "Fetching the latest public events…";
    const events = await github.listPublicEvents();
    // Counting costs no API call, so users classified earlier get their new events too.
    activityState = addActivity(activityState, events);
    render();
    const actors = newActors(events, known, limit);
    if (actors.length === 0) {
      ui.runStatus.textContent = `No new actors among the latest ${events.length} events; GitHub refreshes them every few minutes.`;
      return;
    }
    ui.runStatus.textContent = "Waiting for the model…";
    const { classify } = await classifierPromise;
    let settled = 0;
    ui.runStatus.textContent = `Classifying 0/${actors.length}…`;
    // The first rate-limit error aborts the other lookups; actors not looked up stay unseen for the next refresh.
    await forEachConcurrently(actors, PARALLEL_REQUESTS, async (actor, signal) => {
      const user = await github.getUser(actor.login, signal);
      known.add(actor.login);
      settled += 1;
      ui.runStatus.textContent = `Classifying ${settled}/${actors.length}…`;
      if (user === null) return;
      const text = describeCountryInput(user);
      const [prediction] = await classify([text]);
      if (prediction === undefined) return;
      results.unshift({ ...actor, avatarUrl: user.avatar_url, htmlUrl: user.html_url, text, prediction });
      added += 1;
      render();
    });
    ui.runStatus.textContent = `Added ${plural(added, "user")} from the latest ${events.length} events.`;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    ui.runStatus.textContent = error instanceof RateLimitError ? `${message}. Added ${plural(added, "user")}; a token raises the limit.` : `Error: ${message}`;
  } finally {
    running = false;
    ui.refresh.disabled = false;
  }
}

const followToken = () => {
  ui.limit.value = String(nextProfileLimit(Number(ui.limit.value), ui.token.value.trim() !== ""));
};
ui.token.value = sessionStorage.getItem(TOKEN_KEY) ?? "";
followToken();
ui.token.addEventListener("input", followToken);
ui.toolbar.addEventListener("submit", event => {
  event.preventDefault();
  void refresh();
});
ui.confidence.addEventListener("input", render);
ui.clear.addEventListener("click", () => {
  results.length = 0;
  known.clear();
  activityState = { activity: new Map(), seenEventIds: new Set() };
  selected = null;
  ui.runStatus.textContent = "";
  render();
});

render();
createWorldMap(ui.map, select).then(
  map => {
    worldMap = map;
    render();
  },
  (error: unknown) => {
    ui.summary.textContent = `The map failed to load: ${error instanceof Error ? error.message : String(error)}`;
  },
);
classifierPromise.catch(() => undefined);
void refresh();
