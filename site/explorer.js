// Explorer: any two metrics of one model's runs plotted against each other, one curve per run.
'use strict';

// Scale of the totals (output TPS, power): as measured on the run's n devices, or per 8-card server (× 8 / n).
let SCALE8 = false;
const scaled = (v, r) => (v === null || v === undefined ? v : SCALE8 ? v * 8 / r.n : v);
const perServer = (s) => s + (SCALE8 ? ' · per 8-card server' : '');
const AXES = {
  tps_user: { label: () => 'TPS / User', axis: 'TPS per user (interactivity)', f: (p) => p.out / p.conc },
  out: { label: () => perServer('Output TPS'), f: (p, r) => scaled(p.out, r) },
  total: { label: () => perServer('Total TPS'), f: (p, r) => scaled(p.total, r) },  // input + output tokens
  tps_kw: { label: () => 'Tokens per second (TPS) per kW', title: 'Token Throughput per Kilowatt', f: (p, r) => { const kw = SERVER_KW[family(r.hw)]; return kw ? p.out / r.n * 8 / kw : null; } },
  conc: { label: () => 'Concurrency', f: (p) => p.conc },
  ttft50: { label: () => 'TTFT p50 (s)', f: (p) => p.ttft50 },
  ttft99: { label: () => 'TTFT p99 (s)', f: (p) => p.ttft99 },
  tpot50: { label: () => 'TPOT p50 (ms)', f: (p) => p.tpot50 },
  tpot99: { label: () => 'TPOT p99 (ms)', f: (p) => p.tpot99 },
  tpw: { label: () => 'Output TPS / W', f: (p) => (p.power > 0 ? p.out / p.power : null) },
  power: { label: () => perServer('Power (W)'), f: (p, r) => scaled(p.power, r) },
  custom: { label: () => 'Custom…', f: () => null },  // replaced by the formula of the page state on render
};
// X: what is swept or held to an SLO. Y: what comes out of it.
const X_KEYS = ['tps_user', 'conc'];
const Y_KEYS = ['tps_kw', 'out', 'total', 'tpw', 'power', 'ttft50', 'ttft99', 'tpot50', 'tpot99', 'custom'];
// Y metrics where a taller RNGD curve means RNGD is better, so a ×N multiplier reads naturally.
const HIGHER_IS_BETTER = { out: 1, tps_kw: 1, tpw: 1, custom: 1 };
const W = 900, H = 490, L = 70, R = 880, T = 20, B = 440;  // chart viewBox and plot area

// Series hue of each hardware (style.css --c-<hue>-0..3): RNGD in the Furiosa Core reds (orange at its other
// device counts), each GPU its own hue.
const HUES = [[/RNGD/, 'red'], [/H100/, 'lilac'], [/H200/, 'blue'], [/RTX/, 'green']];
function hueOf(hw) { const h = HUES.find(([re]) => re.test(hw)); return h ? h[1] : 'yellow'; }
function hwName(hw) { const f = family(hw); return (f === 'rngd' ? 'FURIOSA ' : f === 'other' ? '' : 'NVIDIA ') + hw.replace(/-/g, ' '); }

// ---- custom Y formula ----------------------------------------------------------------------------
// Arithmetic over a point's values and per hardware × count parameters, e.g. "out / n * 8 / a".
// Parsed by hand, never eval'd: the formula comes from the URL, so a shared link must not run code.
const POINT_VARS = {
  out: (p) => p.out, conc: (p) => p.conc, n: (p, r) => r.n, isl: (p) => p.isl, osl: (p) => p.osl,
  ttft50: (p) => p.ttft50, ttft99: (p) => p.ttft99, tpot50: (p) => p.tpot50, tpot99: (p) => p.tpot99, power: (p) => p.power,
};
const FUNCS = { min: Math.min, max: Math.max, log: Math.log, log10: Math.log10, sqrt: Math.sqrt, abs: Math.abs, exp: Math.exp };

// Returns { f(vars), params } or throws an Error naming what is wrong.
function compileFormula(src) {
  const toks = String(src).match(/\d*\.?\d+(?:e[+-]?\d+)?|[A-Za-z_]\w*|\S/gi) || [];
  const params = [];
  let i = 0;
  const peek = () => toks[i], take = (t) => { if (t !== undefined && toks[i] !== t) throw new Error('expected ' + t + (toks[i] ? ' before ' + toks[i] : ' at the end')); return toks[i++]; };
  function atom() {
    const t = take();
    if (t === undefined) throw new Error('formula ends too early');
    if (t === '(') { const e = sum(); take(')'); return e; }
    if (t === '-') { const e = power(); return (v) => -e(v); }
    if (/^\d|^\./.test(t)) { const c = +t; return () => c; }
    if (!/^[A-Za-z_]/.test(t)) throw new Error('unexpected ' + t);
    if (Object.hasOwn(FUNCS, t)) {
      take('(');
      const args = [sum()];
      while (peek() === ',') { take(','); args.push(sum()); }
      take(')');
      return (v) => FUNCS[t](...args.map((a) => a(v)));
    }
    if (!Object.hasOwn(POINT_VARS, t) && params.indexOf(t) < 0) params.push(t);
    return (v) => v[t];
  }
  function power() { const b = atom(); if (peek() === '^') { take('^'); const e = power(); return (v) => Math.pow(b(v), e(v)); } return b; }
  function product() {
    let e = power();
    while (peek() === '*' || peek() === '/') { const op = take(), l = e, r = power(); e = op === '*' ? (v) => l(v) * r(v) : (v) => l(v) / r(v); }
    return e;
  }
  function sum() {
    let e = product();
    while (peek() === '+' || peek() === '-') { const op = take(), l = e, r = product(); e = op === '+' ? (v) => l(v) + r(v) : (v) => l(v) - r(v); }
    return e;
  }
  const f = sum();
  if (i < toks.length) throw new Error('unexpected ' + toks[i]);
  return { f: f, params: params };
}
function paramKey(name, r) { return 'p.' + name + '.' + r.hw + '.' + r.n; }  // state key of a parameter cell
function niceTicks(max) {
  if (!(max > 0)) return { top: 1, list: [0, 1] };
  const raw = max / 6, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p, top = Math.ceil(max / step) * step, list = [];
  for (let v = 0; v <= top + step / 1000; v += step) list.push(+v.toFixed(6));
  return { top: top, list: list };
}
function smoothPath(pts) {
  let d = 'M ' + pts[0].x.toFixed(1) + ' ' + pts[0].y.toFixed(1);
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    d += ' C ' + [p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6, p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6, p2.x, p2.y].map((n) => n.toFixed(1)).join(' ');
  }
  return d;
}
function yAtX(vals, x) {
  const s = vals.slice().sort((a, b) => a.x - b.x);
  for (let i = 0; i < s.length - 1; i++) if (s[i].x <= x && x <= s[i + 1].x) return s[i].y + (s[i + 1].y - s[i].y) * (x - s[i].x) / ((s[i + 1].x - s[i].x) || 1);
  return null;
}

function downloads(chart) {
  return '<div class="downloads"><span class="faint">Download</span><button data-download="svg" data-chart="' + chart + '">SVG</button><button data-download="png" data-chart="' + chart + '">PNG</button></div>';
}

let SHOWN = [];  // run ids drawn in the last render, for the series checkboxes
let HOVER = null;  // what the last render plotted, for the readout under the pointer

// Pointer readout on the chart: a vertical line at the pointer's x and each curve's value there
// (linear between measured batches), with the nearest measured batch. Dragging works the same way.
function hoverAt(e) {
  const svg = e.target.closest && e.target.closest('svg.chart'), g = svg && svg.querySelector('g.hover'), tip = svg && svg.parentNode.querySelector('.hover-tip');
  document.querySelectorAll('svg.chart g.hover').forEach((x) => { if (x !== g) x.innerHTML = ''; });
  document.querySelectorAll('.hover-tip').forEach((x) => { if (x !== tip) x.hidden = true; });
  if (!g || !HOVER) return;
  const pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const q = pt.matrixTransform(svg.getScreenCTM().inverse()), h = HOVER;
  if (q.x < L || q.x > R || q.y < T || q.y > B) { g.innerHTML = ''; tip.hidden = true; return; }
  const x = (q.x - L) / (R - L) * h.xtop, py = (y) => B - y / h.ytop * (B - T);
  let marks = '<line x1="' + q.x + '" x2="' + q.x + '" y1="' + T + '" y2="' + B + '" stroke="var(--muted)" stroke-width="1"/>', rows = '';
  h.curves.forEach((c) => {
    const y = yAtX(c.vals, x);
    const near = c.vals.reduce((b, v) => (!b || Math.abs(v.x - x) < Math.abs(b.x - x) ? v : b), null);
    if (y !== null) marks += '<circle cx="' + q.x + '" cy="' + py(y) + '" r="4.5" fill="' + c.color + '" stroke="var(--panel)" stroke-width="2"/>';
    rows += '<tr><td><span style="color:' + c.color + '">●</span> ' + esc(c.label) + '</td><td class="num">' + (y === null ? '—' : fmt(y)) + '</td>'
      + '<td class="num faint">' + (near ? 'b' + near.p.conc + ' ' + fmt(near.y) : '') + '</td></tr>';
  });
  g.innerHTML = marks;
  tip.innerHTML = '<div class="muted">' + esc(h.xLabel) + ' <b>' + fmt(x) + '</b></div><table><tr class="faint"><td></td><td class="num">' + esc(h.yLabel)
    + '</td><td class="num">nearest batch</td></tr>' + rows + '</table>';
  tip.hidden = false;
  const box = svg.parentNode.getBoundingClientRect(), left = e.clientX - box.left + 16;
  tip.style.left = (left + tip.offsetWidth > box.width ? e.clientX - box.left - tip.offsetWidth - 16 : left) + 'px';
  tip.style.top = Math.max(0, e.clientY - box.top - tip.offsetHeight / 2) + 'px';
}
// Download the chart as a standalone SVG or a 2× PNG: title and legend drawn in, every colour and font
// resolved from the page's CSS so the file looks the same in Figma, slides or Slack.
function solidColor(css) {
  const ctx = document.createElement('canvas').getContext('2d'), probe = document.createElement('span');
  probe.style.color = css; document.body.appendChild(probe);
  ctx.fillStyle = getComputedStyle(probe).color; probe.remove();
  return ctx.fillStyle;  // canvas normalises any colour to #rrggbb (or rgba)
}
const EXPORTS = {};  // chart name -> { title, subtitle, legend, file, w, h } from the last render
function chartFile(name) {
  const src = document.querySelector('svg.chart[data-chart="' + name + '"]'), h = EXPORTS[name], head = 70 + 24 * Math.ceil(h.legend.length / 2);
  const clone = src.cloneNode(true), from = src.querySelectorAll('*'), to = clone.querySelectorAll('*');
  from.forEach((el, i) => {
    const cs = getComputedStyle(el), s = to[i].style;
    ['fill', 'stroke', 'stop-color'].forEach((k) => {
      const css = cs.getPropertyValue(k), v = !css || css === 'none' ? 'none' : css.startsWith('url(') ? css.replace(/"/g, "'") : solidColor(css);
      s.setProperty(k, v); if (to[i].hasAttribute(k)) to[i].setAttribute(k, v);
    });
    ['stroke-width', 'stroke-dasharray', 'font-family', 'font-size', 'font-weight', 'letter-spacing'].forEach((k) => s.setProperty(k, cs.getPropertyValue(k)));
  });
  clone.querySelectorAll('g.hover').forEach((g) => g.remove());
  const font = getComputedStyle(document.body).fontFamily.replace(/"/g, "'"), mono = getComputedStyle(src.querySelector('text')).fontFamily.replace(/"/g, "'");
  const txt = (x, y, s, size, fill, fam, extra) => '<text x="' + x + '" y="' + y + '" font-family="' + (fam || font) + '" font-size="' + size + '" fill="' + fill + '"' + (extra || '') + '>' + esc(s) + '</text>';
  const c = (v) => solidColor('var(' + v + ')');
  let out = '<svg xmlns="http://www.w3.org/2000/svg" width="' + h.w + '" height="' + (h.h + head) + '" viewBox="0 0 ' + h.w + ' ' + (h.h + head) + '">'
    + '<rect width="100%" height="100%" fill="' + c('--panel') + '"/>' + txt(20, 32, h.title, 18, c('--title')) + txt(20, 54, h.subtitle, 13, c('--muted'));
  h.legend.forEach((l, i) => {
    const x = 20 + 440 * (i % 2), y = 82 + 24 * Math.floor(i / 2);  // two per row: an RNGD entry is long
    out += '<line x1="' + x + '" y1="' + (y - 4) + '" x2="' + (x + 20) + '" y2="' + (y - 4) + '" stroke="' + solidColor(l.color) + '" stroke-width="3" stroke-linecap="round" stroke-dasharray="' + l.dash + '"/>'
      + '<text x="' + (x + 28) + '" y="' + y + '" font-family="' + font + '" font-size="12" fill="' + c('--text') + '"><tspan font-weight="600">' + esc(l.name)
      + '</tspan><tspan dx="8" font-family="' + mono + '" font-size="11" fill="' + c('--muted') + '">' + esc(l.ver) + '</tspan></text>';
  });
  clone.setAttribute('x', 0); clone.setAttribute('y', head); clone.setAttribute('width', h.w); clone.setAttribute('height', h.h);
  return out + new XMLSerializer().serializeToString(clone) + '</svg>';
}
// An SVG drawn as an image cannot reach the page's web fonts (a theme's, e.g. ABC Favorit), so the PNG gets
// the @font-face rules inlined as data URLs. The SVG file keeps the font names, for tools with the fonts installed.
let FONT_CSS = null;
async function inlineFonts() {
  if (FONT_CSS === null) {
    const rules = [];
    for (const sheet of document.styleSheets) {
      let list;
      try { list = sheet.cssRules; } catch (e) { continue; }  // another origin's sheet
      for (const r of list) if (r instanceof CSSFontFaceRule) rules.push(r);
    }
    FONT_CSS = (await Promise.all(rules.map(async (r) => {
      const m = r.cssText.match(/url\("?([^")]+)"?\)/);
      if (!m) return '';
      const blob = await (await fetch(new URL(m[1], r.parentStyleSheet.href))).blob();
      const data = await new Promise((ok) => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.readAsDataURL(blob); });
      return r.cssText.replace(m[0], 'url("' + data + '")');
    }))).join('\n');
  }
  return FONT_CSS;
}
async function pngOf(svg, w) {
  const css = await inlineFonts(), img = new Image(), cv = document.createElement('canvas'), h = +svg.match(/height="(\d+)"/)[1];
  await new Promise((ok) => { img.onload = ok; img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg.replace('>', '><style>' + css + '</style>')); });
  cv.width = w * 2; cv.height = h * 2;
  cv.getContext('2d').drawImage(img, 0, 0, cv.width, cv.height);
  return new Promise((ok) => cv.toBlob(ok, 'image/png'));
}
async function download(kind, chart) {
  const svg = chartFile(chart), a = document.createElement('a');
  a.href = URL.createObjectURL(kind === 'svg' ? new Blob([svg], { type: 'image/svg+xml' }) : await pngOf(svg, EXPORTS[chart].w));
  a.download = EXPORTS[chart].file + '.' + kind;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
if (typeof document !== 'undefined') {
  document.addEventListener('click', (e) => { const b = e.target.closest && e.target.closest('[data-download]'); if (b) download(b.dataset.download, b.dataset.chart); });
  document.addEventListener('pointermove', hoverAt);
}

PAGES.explorer = async function (st) {
  const gen = await generationRuns();
  const models = [...new Set(gen.map((r) => r.model))].sort((a, b) => a.localeCompare(b));
  const model = models.indexOf(st.model) >= 0 ? st.model : models[0];
  const mRuns = gen.filter((r) => r.model === model);
  const tasks = Object.keys(GEN_TASKS).filter((t) => mRuns.some((r) => r.task === t));
  const task = tasks.indexOf(st.task) >= 0 ? st.task : tasks[0];
  const tRuns = mRuns.filter((r) => r.task === task);
  const scens = shapeOptions(tRuns);
  const scen = pickShape(st.scen, scens);
  SCALE8 = st.scale === '8card';
  const xKey = X_KEYS.indexOf(st.x) >= 0 ? st.x : 'tps_user', yKey = Y_KEYS.indexOf(st.y) >= 0 ? st.y : 'tps_kw';
  const slo = st.slo > 0 ? st.slo : 20;
  if (!scen) return '<div class="status">No fixed-length generation results.</div>';
  const [isl, osl] = scen.split('/').map(Number);

  await loadPoints(tRuns);
  const all = tRuns.map((r) => ({ r: r, pts: pointsAt(r, isl, osl), label: seriesLabel(r) }))
    .filter((o) => o.pts.length)
    .sort((a, b) => famOrder(a.r.hw) - famOrder(b.r.hw) || a.r.n - b.r.n || cmpVer(b.r.ver, a.r.ver));
  // The newest version measured here gets the strong colour, older ones the light one.
  const newestHere = newestVersions(all.map((o) => o.r));
  // Colour: a hue per hardware with its palette step by SDK version (newest first). RNGD is red at its
  // smallest device count here and orange at the others; a GPU's larger counts are darkened toward the panel.
  // Line: the newest SDK version solid, older ones dashed, finer the older; device counts do not change it.
  const uniq = (xs) => xs.filter((x, i) => xs.indexOf(x) === i);
  all.forEach((o) => {
    const same = all.filter((x) => x.r.hw === o.r.hw);
    const vers = uniq(same.map((x) => x.r.ver)).sort((a, b) => cmpVer(b, a)), ns = uniq(same.map((x) => x.r.n)).sort((a, b) => a - b);
    const rngd = family(o.r.hw) === 'rngd', rank = ns.indexOf(o.r.n), shade = rngd ? Math.max(0, rank - 1) : rank;
    const base = 'var(--c-' + (rngd && rank ? 'orange' : hueOf(o.r.hw)) + '-' + Math.min(3, vers.indexOf(o.r.ver)) + ')';
    o.dash = ['none', '6 4', '2 4', '10 4 2 4'][Math.min(3, vers.indexOf(o.r.ver))];
    o.color = shade ? 'color-mix(in srgb, ' + base + ' ' + Math.max(40, 100 - 25 * shade) + '%, var(--panel))' : base;
  });
  // Shown by default: the newest RNGD release at its smallest device count, and each GPU's newest
  // framework at the device count closest to it. The checkboxes override this (state.on).
  let rngdN = null;
  all.forEach((o) => { if (family(o.r.hw) === 'rngd' && o.r.ver === newestHere[o.r.hw] && (rngdN === null || o.r.n < rngdN)) rngdN = o.r.n; });
  const nearestN = {};
  all.forEach((o) => {
    if (o.r.ver !== newestHere[o.r.hw]) return;
    const cur = nearestN[o.r.hw], d = (n) => Math.abs(n - (rngdN === null ? 1 : rngdN));
    if (cur === undefined || d(o.r.n) < d(cur) || (d(o.r.n) === d(cur) && o.r.n < cur)) nearestN[o.r.hw] = o.r.n;
  });
  // A selection left over from another model or task matches nothing here: fall back to the defaults.
  const on = st.on && all.some((o) => st.on.split('~').indexOf(o.r.id) >= 0) ? st.on.split('~') : null;
  all.forEach((o) => { o.on = on ? on.indexOf(o.r.id) >= 0 : o.r.ver === newestHere[o.r.hw] && o.r.n === nearestN[o.r.hw]; });
  SHOWN = all.filter((o) => o.on).map((o) => o.r.id);
  const visible = all.filter((o) => o.on);

  // Custom Y: the formula, its parameters per hardware × count from the state, and any error to show.
  const formula = st.fy || 'out / n * 8 / a', fname = st.fname || 'Custom';
  let custom = { f: null, params: [], error: '' };
  if (yKey === 'custom') {
    try { custom = Object.assign(custom, compileFormula(formula)); } catch (e) { custom.error = e.message; }
  }
  const my = yKey !== 'custom' ? AXES[yKey] : {
    label: () => fname,
    f: (p, r) => {
      if (!custom.f) return null;
      const v = {};
      Object.keys(POINT_VARS).forEach((k) => { v[k] = POINT_VARS[k](p, r); });
      for (const k of custom.params) { const x = st[paramKey(k, r)]; if (x === undefined || x === '' || !isFinite(+x)) return null; v[k] = +x; }
      return custom.f(v);
    },
  };
  const mx = AXES[xKey];
  // Failed points stay off the chart (the batch tables grey them).
  let xmax = 0, ymax = 0;
  const plotted = visible.map((o) => {
    const vals = [];
    o.pts.forEach((p) => {
      const x = p.ok ? mx.f(p, o.r) : null, y = p.ok ? my.f(p, o.r) : null;
      if (x === null || y === null || !isFinite(x) || !isFinite(y)) return;
      xmax = Math.max(xmax, x); ymax = Math.max(ymax, y);
      vals.push({ p: p, x: x, y: y });
    });
    return { o: o, vals: vals };
  });
  const sloOnX = xKey === 'tps_user';  // the SLO is a TPS/user, so it only means something on that axis
  if (sloOnX) xmax = Math.max(xmax, slo);
  const xt = niceTicks(xmax * 1.02), yt = niceTicks(ymax * 1.05);
  const px = (x) => L + x / xt.top * (R - L), py = (y) => B - y / yt.top * (B - T);

  let svg = '<line x1="' + L + '" y1="' + B + '" x2="' + R + '" y2="' + B + '" stroke="var(--faint)"/><line x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + B + '" stroke="var(--faint)"/>';
  xt.list.forEach((v) => { svg += '<text x="' + px(v) + '" y="' + (B + 20) + '" text-anchor="middle">' + fmt(v) + '</text>'; });
  yt.list.forEach((v) => { svg += '<text x="' + (L - 8) + '" y="' + (py(v) + 4) + '" text-anchor="end">' + fmt(v) + '</text>'; });
  const axisTitle = (a) => (a.axis || a.label()).toUpperCase().replace(/KW\b/g, 'kW');
  svg += '<text class="axis-title" x="' + (L + R) / 2 + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(axisTitle(mx)) + '</text>';
  svg += '<text class="axis-title" transform="translate(14 ' + (T + B) / 2 + ') rotate(-90)" text-anchor="middle">' + esc(axisTitle(my)) + '</text>';
  if (sloOnX) svg += '<line x1="' + px(slo) + '" x2="' + px(slo) + '" y1="' + T + '" y2="' + B + '" stroke="var(--faint)" stroke-dasharray="2 4"/><text x="' + (px(slo) + 6) + '" y="' + (B - 8) + '">SLO ' + slo + '</text>';

  // Curves are named in the legend above the chart; every measured point carries its batch size (b = concurrency).
  plotted.forEach((g) => {
    if (g.vals.length > 1) svg += '<path d="' + smoothPath(g.vals.map((v) => ({ x: px(v.x), y: py(v.y) }))) + '" fill="none" stroke="' + g.o.color + '" stroke-width="2.5" stroke-dasharray="' + g.o.dash + '"/>';
    g.vals.forEach((v) => {
      const p = v.p, tip = g.o.label + ' · b' + p.conc + '\nTPS/User ' + fmt(p.out / p.conc) + '\n' + perServer('Output TPS') + ' ' + fmt(scaled(p.out, g.o.r)) + '\n' + perServer('Total TPS') + ' ' + fmt(scaled(p.total, g.o.r))
        + '\nTPOT p50 ' + fmt(p.tpot50) + ' ms\nTTFT p50 ' + fmt(p.ttft50) + ' s' + (yKey === 'custom' ? '\n' + fname + ' ' + fmt(v.y) : '');
      svg += '<circle cx="' + px(v.x) + '" cy="' + py(v.y) + '" r="3.5" fill="' + g.o.color + '" stroke="var(--panel)" stroke-width="1.5"><title>' + esc(tip) + '</title></circle>';
      svg += '<text class="conc" x="' + (px(v.x) + 6) + '" y="' + (py(v.y) - 8) + '" style="fill:' + g.o.color + '">b' + p.conc + '</text>';
    });
  });

  // RNGD-vs-GPU multiplier at the SLO: an arrow from the chosen GPU curve up (or down) to the RNGD curve.
  const rngdG = plotted.filter((g) => family(g.o.r.hw) === 'rngd')[0], gpuGs = plotted.filter((g) => family(g.o.r.hw) !== 'rngd');
  const vsG = gpuGs.filter((g) => g.o.r.id === st.vs)[0];  // none picked (the default): no arrow
  const showVs = sloOnX && !!HIGHER_IS_BETTER[yKey] && !!rngdG && gpuGs.length > 0;
  let vsMissing = '';  // the picked comparison has no value at the SLO: said in the caption instead of an arrow
  if (showVs && vsG) {
    const yr = yAtX(rngdG.vals, slo), yg = yAtX(vsG.vals, slo);
    if (yr === null || yg === null) vsMissing = ' No RNGD vs ' + vsG.o.label + ' multiplier: ' + (yr === null ? 'RNGD' : vsG.o.label) + ' was not measured at ' + slo + ' TPS per user.';
    if (yr !== null && yg !== null && yg > 0) {
      const X = px(slo), Pr = py(yr), Pg = py(yg), up = Pr < Pg, head = up ? Pr + 1 : Pr - 1, dir = up ? 6 : -6;
      svg += '<path d="M ' + X + ' ' + Pg + ' L ' + X + ' ' + head + ' M ' + (X - 4) + ' ' + (head + dir) + ' L ' + X + ' ' + head + ' L ' + (X + 4) + ' ' + (head + dir) + '" fill="none" stroke="var(--muted)" stroke-width="1.2"/>'
        + '<text class="mult" x="' + X + '" y="' + (up ? Pr - 12 : Pr + 22) + '" text-anchor="middle">' + (yr / yg).toFixed(1) + '×</text>';
    }
  }

  // Batch tables, one per Y metric: a row per batch (concurrency), a column per series shown.
  const concs = [...new Set(visible.flatMap((o) => o.pts.map((p) => p.conc)))].sort((a, b) => a - b);
  const colHead = visible.map((o) => '<th class="num"><span style="color:' + o.color + '">●</span> ' + esc(hwName(o.r.hw) + ' ×' + o.r.n)
    + '<div class="faint mono">' + esc(backendLabel(o.r.backend) + ' ' + o.r.ver) + '</div></th>').join('');
  // The metric is picked from tiles; until the viewer picks one, the table follows the Y axis.
  const metricOf = (k) => (k === yKey ? my : AXES[k]);
  const tItems = Y_KEYS.filter((k) => k !== 'custom' || yKey === 'custom').map((k) => ({ key: k, name: metricOf(k).title || metricOf(k).label(), sub: k === yKey ? 'Y axis' : '' }));
  const tKey = pickedKey(st, 'table', tItems, yKey), tm = metricOf(tKey);
  const tRows = concs.map((c) => '<tr><td class="num">b' + c + '</td>' + visible.map((o) => {
    const p = o.pts.filter((x) => x.conc === c)[0], v = p ? tm.f(p, o.r) : null;
    return '<td class="num' + (p && !p.ok ? ' faint" title="' + esc(p.raw.fail_reason || 'failed') : '') + '">' + fmt(v) + '</td>';
  }).join('') + '</tr>').join('');
  const tables = pickTiles('table', tItems, tKey) + '<div class="picked"><h3>' + esc(tm.title || tm.label()) + '</h3>'
    + '<div class="scroll"><table class="list"><thead><tr><th class="num">Batch</th>' + colHead + '</tr></thead><tbody>' + tRows + '</tbody></table></div></div>';
  HOVER = {
    xtop: xt.top, ytop: yt.top, xLabel: mx.label(), yLabel: my.label(),
    curves: plotted.map((g) => ({ label: hwName(g.o.r.hw) + ' ×' + g.o.r.n + ' · ' + g.o.r.ver, color: g.o.color, vals: g.vals })).filter((c) => c.vals.length),
  };

  const seriesList = all.map((o) => '<label class="series"><input type="checkbox" data-toggle="' + esc(o.r.id) + '"' + (o.on ? ' checked' : '') + '>'
    + '<svg width="28" height="10" aria-hidden="true"><line x1="0" y1="5" x2="28" y2="5" stroke="' + o.color + '" stroke-width="2" stroke-dasharray="' + o.dash + '"/></svg><span>' + esc(o.label) + '</span></label>').join('');
  const axisOptions = (keys) => keys.map((k) => ({ v: k, label: AXES[k].label() }));
  // Each legend entry opens how that run was measured (run.js).
  const legend = '<div class="legend chart-legend">' + visible.map((o) => '<a class="run-link" title="How this run was measured" href="#run?id=' + encodeURIComponent(o.r.id) + '&scen=' + encodeURIComponent(scen) + '"><svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="' + o.color
    + '" stroke-width="3" stroke-linecap="round" stroke-dasharray="' + o.dash + '"/></svg><b>' + esc(hwName(o.r.hw) + ' ×' + o.r.n + (o.r.layout ? ' ' + o.r.layout : ''))
    + '</b><span class="mono">' + esc(backendLabel(o.r.backend) + ' ' + o.r.ver) + '</span><span class="faint">↗</span></a>').join('') + '</div>';
  const caption = model + ', ' + isl + '/' + osl + ' ISL/OSL. ' + visible.map((o) => o.label).join(', ') + '.'
    + (yKey === 'tps_kw' ? ' Per-device throughput scaled to 8-card server power: RNGD 3.0 kW, RTX PRO 6000 6.5 kW, H100/H200 10.2 kW.' : '')
    + (yKey === 'custom' ? ' y = ' + formula + '.' : '') + vsMissing
    + (SCALE8 ? ' Output TPS, total TPS and power are scaled to an 8-card server (× 8 / devices measured).' : '');
  const subtitle = model + ' · ISL ' + isl + ' / OSL ' + osl + ' · ' + GEN_TASKS[task].toLowerCase(), fileBase = [model, isl + '-' + osl].join('_');
  EXPORTS.main = {
    w: W, h: H, title: my.title || my.label(), subtitle: subtitle, file: (fileBase + '_' + yKey + '_vs_' + xKey).replace(/[^\w.-]+/g, '-'),
    legend: visible.map((o) => ({ color: o.color, dash: o.dash, name: hwName(o.r.hw) + ' ×' + o.r.n, ver: backendLabel(o.r.backend) + ' ' + o.r.ver })),
  };
  // Rack capacity at the SLO (usersPerRack): whole 8-card servers per rack × users per server.
  const rackKw = +st.rack > 0 ? +st.rack : RACK_KW;
  const racks = visible.map((o) => { const x = usersPerRack(o.pts, o.r, slo, rackKw); return x && Object.assign(x, { o: o, users: x.v }); }).filter(Boolean);  // in legend order
  const rRack = racks.filter((x) => family(x.o.r.hw) === 'rngd')[0];
  const gRack = racks.filter((x) => family(x.o.r.hw) !== 'rngd').sort((a, b) => b.users - a.users)[0];  // the best GPU
  // Vertical bars, users per rack on Y. A lower bound (≥) fades out at its top.
  const RW = 900, RH = 380, rL = 70, rR = 880, rT = 30, rB = 290, yr = niceTicks(Math.max(...racks.map((x) => x.users), 1) * 1.12);
  const ry = (v) => rB - v / yr.top * (rB - rT), slot = (rR - rL) / Math.max(racks.length, 1), bw = Math.min(80, slot * 0.5);
  let rsvg = '<defs>' + racks.map((x, i) => '<linearGradient id="rg' + i + '" x1="0" y1="1" x2="0" y2="0"><stop offset="0.7" style="stop-color:' + x.o.color + '"/><stop offset="1" style="stop-color:' + x.o.color + ';stop-opacity:0.15"/></linearGradient>').join('') + '</defs>';
  yr.list.forEach((v) => { rsvg += '<line x1="' + rL + '" x2="' + rR + '" y1="' + ry(v) + '" y2="' + ry(v) + '" stroke="var(--grid)"/><text x="' + (rL - 8) + '" y="' + (ry(v) + 4) + '" text-anchor="end">' + fmt(v) + '</text>'; });
  rsvg += '<text class="axis-title" transform="translate(14 ' + (rT + rB) / 2 + ') rotate(-90)" text-anchor="middle">USERS PER RACK AT SLO</text>';
  racks.forEach((x, i) => {
    const cx = rL + slot * (i + 0.5), y = ry(x.users), name = hwName(x.o.r.hw) + ' ×' + x.o.r.n;
    rsvg += '<rect x="' + (cx - bw / 2) + '" y="' + y + '" width="' + bw + '" height="' + (rB - y) + '" rx="2" fill="' + (x.ge ? 'url(#rg' + i + ')' : x.o.color) + '"><title>' + esc(name + ' · ' + x.o.r.ver) + '</title></rect>'
      + '<text class="rack-v" x="' + cx + '" y="' + (y - 8) + '" text-anchor="middle">' + (x.ge ? '≥ ' : '') + fmt(x.users) + '</text>'
      + '<text class="rack-n" x="' + cx + '" y="' + (rB + 20) + '" text-anchor="middle">' + esc(name) + '</text>'
      + '<text x="' + cx + '" y="' + (rB + 36) + '" text-anchor="middle">' + esc(backendLabel(x.o.r.backend) + ' ' + x.o.r.ver) + '</text>'
      + '<text class="rack-s" x="' + cx + '" y="' + (rB + 54) + '" text-anchor="middle">' + fmt(x.servers) + ' srv × ' + (x.ge ? '≥ ' : '') + fmt(x.perServer) + '</text>';
  });
  rsvg += '<line x1="' + rL + '" x2="' + rR + '" y1="' + rB + '" y2="' + rB + '" stroke="var(--faint)"/>';
  EXPORTS.rack = {
    w: RW, h: RH, title: 'Users per rack at SLO', subtitle: subtitle + metricBasis(METRICS.rack, slo, rackKw), legend: [],
    file: (fileBase + '_rack-users_slo' + slo + '_' + rackKw + 'kW').replace(/[^\w.-]+/g, '-'),
  };
  const showRack = st.rackview === 'show';  // off by default: only when the viewer asks for it
  const rackPanel = '<div class="panel rack"><div class="panel-head"><div><h2 style="font-weight:400;font-size:18px">Rack capacity at SLO</h2>'
    + '<div class="muted">Users served at ≥ ' + slo + ' TPS per user by one rack, filled with 8-card servers · ISL ' + isl + ' / OSL ' + osl + '</div></div>'
    + '<div class="controls">' + rackInput(rackKw) + sloInput(slo) + downloads('rack') + '</div></div>'
    + (rRack && gRack && gRack.users > 0 ? '<div class="rack-mult"><b>' + (gRack.ge ? '≤ ' : '') + (rRack.users / gRack.users).toFixed(2) + '×</b> RNGD vs ' + esc(gRack.o.label)
      + (gRack.ge ? ' <span class="faint">(the GPU value is a lower bound)</span>' : '') + '</div>' : '')
    + (racks.length ? '<svg class="chart rack-chart" data-chart="rack" viewBox="0 0 ' + RW + ' ' + RH + '" width="100%" role="img" aria-label="Users per rack at the SLO">' + rsvg + '</svg>' : '<div class="status">No series with a server power and an SLO result.</div>')
    + '<p class="caption">Servers per rack = rack power ÷ 8-card server power (RNGD 3.0 kW, RTX PRO 6000 6.5 kW, H100/H200 10.2 kW assumed), rounded down to whole servers. '
    + 'Users per server = the most concurrent users still at ≥ ' + slo + ' TPS per user, × 8 ÷ devices measured. ≥: the largest batch measured still met the SLO.</p></div>';
  // Custom Y: name, formula, and a parameter table with one row per hardware × count shown.
  const combos = visible.filter((o, i) => visible.findIndex((x) => x.r.hw === o.r.hw && x.r.n === o.r.n) === i);
  const customPanel = yKey !== 'custom' ? '' : '<div class="panel custom-y"><h2 style="font-weight:400;font-size:18px">Custom Y axis</h2>'
    + '<div class="custom-row"><label class="ctl">Name<input data-k="fname" value="' + esc(fname) + '"></label>'
    + '<label class="ctl grow">Formula · y =<input class="mono" data-k="fy" value="' + esc(formula) + '" spellcheck="false"></label>'
    + '<span class="' + (custom.error ? 'fail' : 'ok') + '">' + esc(custom.error || '✓ valid') + '</span></div>'
    + '<div class="chips"><span class="muted">Point values</span>' + Object.keys(POINT_VARS).map((k) => '<code>' + k + '</code>').join('')
    + '<span class="muted">Parameters</span>' + (custom.params.map((k) => '<code class="param">' + esc(k) + '</code>').join('') || '<span class="faint">none</span>')
    + '<span class="muted">Functions</span><code>' + Object.keys(FUNCS).join(' ') + '</code></div>'
    + (custom.params.length ? '<table class="list"><thead><tr><th>Hardware × count</th>' + custom.params.map((k) => '<th>' + esc(k) + '</th>').join('') + '</tr></thead><tbody>'
      + combos.map((o) => '<tr><td><span style="color:' + o.color + '">●</span> ' + esc(hwName(o.r.hw) + ' ×' + o.r.n) + '</td>'
        + custom.params.map((k) => '<td><input type="number" step="any" class="mono" data-k="' + esc(paramKey(k, o.r)) + '" value="' + esc(st[paramKey(k, o.r)] ?? '') + '"></td>').join('') + '</tr>').join('')
      + '</tbody></table><p class="caption">A series with an empty parameter is left out. The formula and parameters are kept in the link.</p>' : '')
    + '</div>';

  return '<div class="explorer"><aside class="panel">'
    + select('Model', 'model', model, models) + select('Task', 'task', task, tasks.map((t) => ({ v: t, label: GEN_TASKS[t] })))
    + select('ISL / OSL', 'scen', scen, scens)
    + '<div class="ctl">Hardware · backend' + seriesList + '</div>'
    + '</aside><section><div class="controls">' + select('X axis', 'x', xKey, axisOptions(X_KEYS)) + select('Y axis', 'y', yKey, axisOptions(Y_KEYS))
    + select('Scale', 'scale', SCALE8 ? '8card' : 'measured', [{ v: 'measured', label: 'As measured' }, { v: '8card', label: 'Per 8-card server' }]) + (sloOnX ? sloInput(slo) : '')
    + (showVs ? select('× RNGD vs', 'vs', vsG ? vsG.o.r.id : 'none', [{ v: 'none', label: 'None' }].concat(gpuGs.map((g) => ({ v: g.o.r.id, label: g.o.label })))) : '')
    + select('Rack capacity', 'rackview', showRack ? 'show' : 'hide', [{ v: 'hide', label: 'Hide' }, { v: 'show', label: 'Show' }]) + '</div>'
    + '<div class="panel"><div class="panel-head"><div><h2 style="font-weight:400;font-size:18px">' + esc(my.title || my.label()) + '</h2><div class="muted">ISL ' + isl + ' / OSL ' + osl + ' · ' + esc(GEN_TASKS[task].toLowerCase()) + '</div></div>'
    + downloads('main') + '</div>' + legend
    + '<div class="chart-wrap"><svg class="chart" data-chart="main" viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="' + esc(my.label() + ' against ' + mx.label() + ' for ' + model) + '">' + svg + '<g class="hover"></g></svg><div class="hover-tip" hidden></div></div>'
    + '<p class="caption">' + esc(caption) + '</p></div>' + (showRack ? rackPanel : '') + customPanel
    + '<div class="panel batches"><div class="panel-head"><h2 style="font-weight:400;font-size:18px">Measured batches</h2><span class="muted">ISL ' + isl + ' / OSL ' + osl + ' · pick a metric, failed points greyed</span></div>'
    + tables + '</div></section></div>';
};

PAGES.explorer.toggle = function (id, checked) {
  const shown = SHOWN.filter((x) => x !== id).concat(checked ? [id] : []);
  setState({ on: shown.join('~') });
};
