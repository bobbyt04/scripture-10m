// SVG charts drawn without a library. Colours come from CSS custom properties (muted book colours, greys).

import { h, s } from "../dom.ts";

export interface Series {
  label: string;
  color: string; // a CSS colour, usually var(--gita) etc.
  points: [number, number][];
  dashed?: boolean;
  width?: number;
}

export interface HLine {
  y: number;
  label: string;
  color: string;
  below?: boolean; // put the label under the line, for lines close together
}

export interface Marker {
  x: number;
  y: number;
  label: string;
  color: string;
}

export interface LineChartSpec {
  title: string;
  series: Series[];
  hlines?: HLine[];
  markers?: Marker[];
  x: { domain: [number, number]; ticks: number[]; format: (v: number) => string; label: string };
  y: { domain: [number, number]; ticks: number[]; format: (v: number) => string; label: string };
}

let uid = 0;

export function lineChart(spec: LineChartSpec, width: number): SVGSVGElement {
  const W = Math.max(300, Math.round(width));
  const H = Math.round(Math.min(380, Math.max(250, W * 0.52)));
  const m = { top: 16, right: 24, bottom: 44, left: 48 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const [x0, x1] = spec.x.domain, [y0, y1] = spec.y.domain;
  const X = (v: number) => m.left + ((v - x0) / (x1 - x0)) * iw;
  const Y = (v: number) => m.top + (1 - (v - y0) / (y1 - y0)) * ih;
  const clipId = `clip-${++uid}`;

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": spec.title });
  svg.append(s("defs", {}, s("clipPath", { id: clipId }, s("rect", { class: "chart__clip", x: m.left - 4, y: 0, width: iw + 8, height: H }))));

  const grid = s("g", { class: "chart__grid" });
  for (const t of spec.y.ticks) {
    grid.append(s("line", { x1: m.left, x2: W - m.right, y1: Y(t), y2: Y(t) }));
    grid.append(s("text", { x: m.left - 8, y: Y(t), "text-anchor": "end", "dominant-baseline": "middle" }, spec.y.format(t)));
  }
  for (const t of spec.x.ticks) {
    grid.append(s("text", { x: X(t), y: H - m.bottom + 18, "text-anchor": "middle" }, spec.x.format(t)));
  }
  grid.append(s("line", { class: "chart__axis", x1: m.left, x2: W - m.right, y1: m.top + ih, y2: m.top + ih }));
  grid.append(s("text", { class: "chart__label", x: m.left + iw / 2, y: H - 6, "text-anchor": "middle" }, spec.x.label));
  grid.append(s("text", { class: "chart__label", x: 12, y: m.top + ih / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.top + ih / 2})` }, spec.y.label));
  svg.append(grid);

  for (const l of spec.hlines ?? []) {
    svg.append(s("g", { class: "chart__hline" },
      s("line", { x1: m.left, x2: W - m.right, y1: Y(l.y), y2: Y(l.y), stroke: l.color }),
      s("text", { x: W - m.right - 4, y: Y(l.y) + (l.below ? 15 : -6), "text-anchor": "end", fill: l.color }, l.label)));
  }

  const lines = s("g", { "clip-path": `url(#${clipId})` });
  for (const ser of spec.series) {
    const d = ser.points
      .filter(([, y]) => y >= y0 && y <= y1)
      .map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
    lines.append(s("path", {
      d, fill: "none", stroke: ser.color, "stroke-width": ser.width ?? 2, "stroke-linejoin": "round", "stroke-linecap": "round",
      "stroke-dasharray": ser.dashed ? "5 5" : null, class: "chart__line",
    }));
  }
  svg.append(lines);

  for (const mk of spec.markers ?? []) {
    svg.append(s("g", { class: "chart__marker" },
      s("circle", { cx: X(mk.x), cy: Y(mk.y), r: 4.5, fill: "var(--bg)", stroke: mk.color, "stroke-width": 2 }),
      s("text", { x: X(mk.x) - 4, y: Y(mk.y) - 12, "text-anchor": "end", fill: mk.color }, mk.label)));
  }
  return svg;
}

export function legend(items: { label: string; color: string; dashed?: boolean; kind?: "line" | "dot" }[]): HTMLElement {
  return h("ul", { class: "legend", role: "list" },
    items.map((i) =>
      h("li", {},
        h("span", { class: `legend__key${i.dashed ? " legend__key--dashed" : ""}${i.kind === "dot" ? " legend__key--dot" : ""}`, style: `--c:${i.color}` }),
        i.label)));
}

/** Re-draw a chart whenever its container changes width noticeably, so text stays a readable size. */
export function responsive(container: HTMLElement, draw: (width: number) => Element): void {
  let last = 0;
  const render = () => {
    const w = container.clientWidth || 640;
    if (Math.abs(w - last) < 24) return;
    last = w;
    container.replaceChildren(draw(w));
  };
  render();
  if (typeof ResizeObserver === "function") new ResizeObserver(render).observe(container);
}

export interface MissRow {
  label: string;
  sub: string;
  missed: number;
  harmful: number;
  /** Drawn apart from the rows above it: a different design was under test. */
  apart?: boolean;
}

/** Crisis messages each review's rules missed (light bar), and how many of those reached a harmful verse (dark). */
export function missChart(rows: MissRow[], width: number, title: string): SVGSVGElement {
  const W = Math.max(300, Math.round(width));
  const narrow = W < 560;
  const labelW = narrow ? 0 : 250;
  const rowH = narrow ? 76 : 54;
  const gap = 16;
  const m = { top: 6, right: 12, bottom: 4, left: 2 };
  const H = m.top + rows.length * rowH + (rows.some((r) => r.apart) ? gap : 0) + m.bottom;
  const max = Math.max(...rows.map((r) => r.missed));
  const x0 = m.left + labelW;
  const span = W - x0 - m.right - (narrow ? 0 : 160);
  const X = (v: number) => (v / max) * span;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart chart--miss", role: "img", "aria-label": title });
  let y = m.top;
  for (const r of rows) {
    if (r.apart) {
      svg.append(s("line", { x1: m.left, x2: W - m.right, y1: y + gap / 2 - 2, y2: y + gap / 2 - 2, class: "miss__sep" }));
      y += gap;
    }
    const g = s("g", { class: "chart__row" });
    const barY = narrow ? y + 34 : y + 10;
    g.append(s("text", { x: m.left, y: narrow ? y + 13 : y + 19, class: "chart__rowlabel miss__label" }, r.label));
    g.append(s("text", { x: m.left, y: narrow ? y + 28 : y + 36, class: "chart__note" }, r.sub));
    g.append(s("rect", { x: x0, y: barY, width: X(r.missed), height: 16, rx: 3, class: "miss__missed" }));
    g.append(s("rect", { x: x0, y: barY, width: Math.max(2, X(r.harmful)), height: 16, rx: 3, class: r.apart ? "miss__harm miss__harm--apart" : "miss__harm" }));
    g.append(s("text", { x: narrow ? x0 : x0 + X(r.missed) + 10, y: narrow ? barY + 32 : barY + 12, class: "chart__note miss__note" },
      `${r.harmful} harmful of ${r.missed} missed`));
    svg.append(g);
    y += rowH;
  }
  return svg;
}

export interface IntervalRow {
  label: string;
  color: string;
  model: { value: number; lo: number; hi: number };
  baseline: { name: string; value: number; lo: number; hi: number };
}

/** Model against the best n-gram baseline per book: a point and its 95% interval on one axis of nats. */
export function intervalChart(rows: IntervalRow[], width: number, domain: [number, number], title: string): SVGSVGElement {
  const W = Math.max(300, Math.round(width));
  const rowH = 64;
  const m = { top: 8, right: 20, bottom: 40, left: 20 };
  const H = m.top + rows.length * rowH + m.bottom;
  const X = (v: number) => m.left + ((v - domain[0]) / (domain[1] - domain[0])) * (W - m.left - m.right);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": title });
  const grid = s("g", { class: "chart__grid" });
  for (let t = Math.ceil(domain[0] * 10) / 10; t <= domain[1] + 1e-9; t += 0.1) {
    const major = Math.abs(Math.round(t * 10) % 2) === 0;
    grid.append(s("line", { x1: X(t), x2: X(t), y1: m.top, y2: H - m.bottom, class: major ? "" : "chart__minor" }));
    if (major) grid.append(s("text", { x: X(t), y: H - m.bottom + 18, "text-anchor": "middle" }, t.toFixed(1)));
  }
  grid.append(s("text", { class: "chart__label", x: W / 2, y: H - 4, "text-anchor": "middle" }, "validation loss, nats per character (lower is better)"));
  svg.append(grid);
  // a label goes right of its interval, or left of it when it would run off the edge
  const note = (text: string, iv: { lo: number; hi: number }, y: number) => {
    const fits = X(iv.hi) + 10 + text.length * 6.6 <= W - 4;
    return s("text", { x: fits ? X(iv.hi) + 10 : X(iv.lo) - 10, y: y + 4, "text-anchor": fits ? "start" : "end", class: "chart__note" }, text);
  };
  rows.forEach((r, i) => {
    const y = m.top + i * rowH;
    const g = s("g", { class: "chart__row" });
    g.append(s("text", { x: m.left, y: y + 14, class: "chart__rowlabel", fill: r.color }, r.label));
    const ym = y + 32, yb = y + 50;
    g.append(s("line", { x1: X(r.model.lo), x2: X(r.model.hi), y1: ym, y2: ym, stroke: r.color, "stroke-width": 3, "stroke-linecap": "round", class: "chart__ci" }));
    g.append(s("circle", { cx: X(r.model.value), cy: ym, r: 5.5, fill: r.color, class: "chart__pt" }));
    g.append(note(`Scripture-10M ${r.model.value.toFixed(3)}`, r.model, ym));
    g.append(s("line", { x1: X(r.baseline.lo), x2: X(r.baseline.hi), y1: yb, y2: yb, stroke: "var(--muted)", "stroke-width": 3, "stroke-linecap": "round", class: "chart__ci" }));
    g.append(s("circle", { cx: X(r.baseline.value), cy: yb, r: 5, fill: "var(--bg)", stroke: "var(--muted)", "stroke-width": 2, class: "chart__pt" }));
    g.append(note(`${r.baseline.name} ${r.baseline.value.toFixed(3)}`, r.baseline, yb));
    svg.append(g);
  });
  return svg;
}
