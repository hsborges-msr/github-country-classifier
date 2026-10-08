import { geoArea, geoCentroid, geoGraticule10, geoNaturalEarth1, geoPath } from "d3-geo";
import type { Feature, Geometry, Polygon } from "geojson";
import countries from "i18n-iso-countries";
import { feature } from "topojson-client";
import type { GeometryCollection, Topology } from "topojson-specification";
import worldUrl from "world-atlas/countries-50m.json?url";

const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 960;
const HEIGHT = 480;

export interface WorldMap {
  /** Recolor the countries and resize the bubbles; `selected` is highlighted. */
  update(counts: ReadonlyMap<string, number>, selected: string | null): void;
}

type CountryFeature = Feature<Geometry, { name: string }>;

function svgElement<K extends keyof SVGElementTagNameMap>(name: K, attributes: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

/** ISO alpha-2 code of a world-atlas feature (ISO numeric id; Kosovo has none). */
function alpha2(country: CountryFeature): string | null {
  if (country.properties.name === "Kosovo") return "XK";
  return country.id === undefined ? null : (countries.numericToAlpha2(String(country.id)) ?? null);
}

/** Centroid of a country's largest polygon, so overseas territories do not pull the bubble into the ocean. */
function mainCentroid(geometry: Geometry): [number, number] {
  if (geometry.type !== "MultiPolygon") return geoCentroid(geometry);
  const polygons = geometry.coordinates.map((coordinates): Polygon => ({ type: "Polygon", coordinates }));
  const largest = polygons.reduce((best, polygon) => (geoArea(polygon) > geoArea(best) ? polygon : best));
  return geoCentroid(largest);
}

/** Light to dark blue on a log scale of the count relative to the largest one. */
function fillFor(count: number, max: number): string {
  if (count === 0) return "";
  const t = max <= 1 ? 1 : Math.log(count) / Math.log(max);
  const mix = (from: number, to: number) => Math.round(from + (to - from) * (0.15 + 0.85 * t));
  return `rgb(${mix(222, 8)}, ${mix(235, 69)}, ${mix(247, 148)})`;
}

/** Draw the world into `svg`; clicking a country (or its bubble) selects it, clicking it again clears the selection. */
export async function createWorldMap(svg: SVGSVGElement, onSelect: (code: string | null) => void): Promise<WorldMap> {
  const topology = (await (await fetch(worldUrl)).json()) as Topology<{ countries: GeometryCollection<{ name: string }> }>;
  const features = feature(topology, topology.objects.countries).features as CountryFeature[];
  const projection = geoNaturalEarth1().fitExtent([[4, 4], [WIDTH - 4, HEIGHT - 4]], { type: "Sphere" });
  const path = geoPath(projection);

  svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
  svg.replaceChildren(
    svgElement("path", { class: "sphere", d: path({ type: "Sphere" }) ?? "" }),
    svgElement("path", { class: "graticule", d: path(geoGraticule10()) ?? "" }),
  );
  const countryLayer = svgElement("g");
  const bubbleLayer = svgElement("g", { class: "bubbles" });
  svg.append(countryLayer, bubbleLayer);

  const names = new Intl.DisplayNames(["en"], { type: "region" });
  const shapes = new Map<string, { path: SVGPathElement; title: SVGTitleElement; centroid: [number, number] | null }>();
  let selected: string | null = null;

  for (const country of features) {
    const code = alpha2(country);
    const element = svgElement("path", { class: "country", d: path(country) ?? "" });
    const title = svgElement("title");
    title.textContent = country.properties.name;
    element.append(title);
    countryLayer.append(element);
    if (code === null) continue;
    element.addEventListener("click", () => onSelect(selected === code ? null : code));
    const centroid = projection(mainCentroid(country.geometry));
    shapes.set(code, { path: element, title, centroid: centroid ? [centroid[0], centroid[1]] : null });
  }

  return {
    update(counts, selection) {
      selected = selection;
      const max = Math.max(1, ...counts.values());
      bubbleLayer.replaceChildren();
      for (const [code, shape] of shapes) {
        const count = counts.get(code) ?? 0;
        shape.path.style.fill = fillFor(count, max);
        shape.path.classList.toggle("selected", code === selection);
        shape.title.textContent = `${names.of(code) ?? code}: ${count} ${count === 1 ? "user" : "users"}`;
        if (count === 0 || shape.centroid === null) continue;
        const bubble = svgElement("circle", { cx: shape.centroid[0], cy: shape.centroid[1], r: 2.5 + 2.5 * Math.sqrt(count), class: code === selection ? "selected" : "" });
        const label = svgElement("title");
        label.textContent = shape.title.textContent;
        bubble.append(label);
        bubble.addEventListener("click", () => onSelect(selected === code ? null : code));
        bubbleLayer.append(bubble);
      }
    },
  };
}
