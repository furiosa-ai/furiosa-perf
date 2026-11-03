// Shared by every page: data loading, the page router, formatting and the metrics the pages compare.
'use strict';

// ---- data ----------------------------------------------------------------------------------------
// data/index.json lists every run; data/runs/<id>.json holds one run's points (loaded on demand).
const DATA = { index: null, points: {}, details: {} };  // details: how each run was measured (run.js)

async function loadIndex() {
  if (!DATA.index) {
    const r = await fetch('data/index.json');
    if (!r.ok) throw new Error('data/index.json: HTTP ' + r.status);
    DATA.index = (await r.json()).runs;
  }
  return DATA.index;
}

// One point of a fixed-length run, with the fields the pages use. `raw` keeps every recorded value.
function toPoint(raw) {
  const out = raw['Output Throughput(tok/s)'];
  return {
    isl: raw['Input Tokens'], osl: raw['Output Tokens'], conc: raw['Concurrent'], out: out, total: raw['Total Throughput(tok/s)'],
    ttft50: raw['Median TTFT(s)'], ttft99: raw['P99 TTFT(s)'], tpot50: raw['Median TPOT(ms)'], tpot99: raw['P99 TPOT(ms)'],
    power: raw['mean_power'],
    // Results of the presets judge each point (ok); older ones only tell a failed point by no output.
    ok: (raw.ok === undefined || !!raw.ok) && out > 0,
    raw: raw,
  };
}

async function loadPoints(runs) {
  await Promise.all(runs.filter((run) => !DATA.points[run.id]).map(async (run) => {
    const r = await fetch('data/runs/' + encodeURIComponent(run.id) + '.json');
    if (!r.ok) throw new Error(run.id + ': HTTP ' + r.status);
    const d = await r.json();
    DATA.points[run.id] = d.rows.map((row) => toPoint(Object.fromEntries(d.cols.map((c, i) => [c, row[i]]))));
    DATA.details[run.id] = d.details || {};
  }));
}

// Points of one ISL/OSL pair, by concurrency.
function pointsAt(run, isl, osl) {
  return (DATA.points[run.id] || []).filter((p) => p.isl === isl && p.osl === osl).sort((a, b) => a.conc - b.conc);
}

// The newest run of each identical measurement (same platform, layout, task, model and server options).
function latestRuns(runs) {
  const best = {};
  runs.forEach((r) => {
    const k = [r.hw, r.n, r.layout, r.backend, r.ver, r.kind, r.task, r.model, r.variant].join('|');
    if (!best[k] || (r.date || '') > (best[k].date || '')) best[k] = r;
  });
  return Object.values(best);
}

// Fixed-length generation runs as configured (no --server-arg): the ones the Summary, Explorer and Releases pages compare.
async function generationRuns() {
  return latestRuns((await loadIndex()).filter((r) => r.kind === 'baseline' && !r.variant && GEN_TASKS[r.task]));
}
const GEN_TASKS = { 'text': 'Text', 'image-text': 'Image + text' };

function shapeOptions(runs) {
  const set = {};
  runs.forEach((r) => (r.shapes || []).forEach((s) => { set[s] = s.split('/').map(Number); }));
  return Object.keys(set).sort((a, b) => set[a][0] - set[b][0] || set[a][1] - set[b][1]);
}
function pickShape(want, options) {
  return options.indexOf(want) >= 0 ? want : options.indexOf('1024/1024') >= 0 ? '1024/1024' : options[0];
}

// ---- platforms -----------------------------------------------------------------------------------
function verKey(v) {
  const m = /^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?/.exec(v) || [];
  const pre = { a: 0, b: 1, rc: 2 }[m[4]];
  return [+m[1] || 0, +m[2] || 0, +m[3] || 0, pre === undefined ? 3 : pre, +m[5] || 0];
}
function cmpVer(a, b) {
  const x = verKey(a), y = verKey(b);
  for (let i = 0; i < 5; i++) if (x[i] !== y[i]) return x[i] - y[i];
  return 0;
}
function family(hw) {
  if (hw.indexOf('RNGD') >= 0) return 'rngd';
  if (hw.indexOf('H200') >= 0) return 'h200';
  if (hw.indexOf('H100') >= 0) return 'h100';
  if (hw.indexOf('RTX') >= 0) return 'rtx';
  return 'other';
}
const FAMILY_ORDER = { rngd: 0, h100: 1, h200: 2, rtx: 3, other: 4 };
function famOrder(hw) { return FAMILY_ORDER[family(hw)]; }
function backendLabel(b) { return b === 'vllm' ? 'vLLM' : b; }
function platformKey(r) { return r.hw + '|' + r.backend + '|' + r.ver; }
function platformLabel(k) { const p = k.split('|'); return p[0] + ' · ' + backendLabel(p[1]) + ' ' + p[2]; }
function seriesLabel(r) { return r.hw + ' ×' + r.n + (r.layout ? ' ' + r.layout : '') + ' · ' + backendLabel(r.backend) + ' ' + r.ver; }
// Newest version measured on each device: its series get the strong colour, older ones the light one.
function newestVersions(runs) {
  const newest = {};
  runs.forEach((r) => { if (!newest[r.hw] || cmpVer(r.ver, newest[r.hw]) > 0) newest[r.hw] = r.ver; });
  return newest;
}

// ---- metrics -------------------------------------------------------------------------------------
// Power of a standard 8-card server, kW: RNGD and RTX PRO 6000 as in the published efficiency chart.
// ponytail: H100/H200 assumed 10.2 kW (HGX), replace with the official figure when there is one.
const SERVER_KW = { rngd: 3.0, rtx: 6.5, h100: 10.2, h200: 10.2 };

function okPts(pts) { return pts.filter((p) => p.ok); }
function peakPoint(pts) { return okPts(pts).reduce((b, p) => (!b || p.out > b.out ? p : b), null); }
function c1(pts) { return okPts(pts).filter((p) => p.conc === 1)[0]; }

// Largest concurrency whose TPS/user is still >= slo, interpolated between measured points.
// ge: the sweep's top concurrency still met it, so the true value is at least that.
function usersAtSlo(pts, slo) {
  const ok = okPts(pts), c = ok.map((p) => p.conc), t = ok.map((p) => p.out / p.conc);
  if (!c.length || !(slo > 0)) return null;
  if (!t.some((v) => v >= slo)) return { v: 0, ge: false };
  if (t[t.length - 1] >= slo) return { v: c[c.length - 1], ge: true };
  let best = null;
  for (let i = 0; i < c.length - 1; i++) {
    const y0 = t[i], y1 = t[i + 1];
    if (y0 !== y1 && (y0 - slo) * (y1 - slo) <= 0) {
      const x = c[i] + (slo - y0) / (y1 - y0) * (c[i + 1] - c[i]);
      best = best === null ? x : Math.max(best, x);
    }
  }
  return { v: best === null ? 0 : best, ge: false };
}

// Tokens per second per kW of an 8-card server, read off the throughput curve where TPS/user = slo.
// A curve faster than the SLO even at its highest concurrency only gives a lower bound.
function tpsPerKwAtSlo(pts, run, slo) {
  const kw = SERVER_KW[family(run.hw)];
  const c = okPts(pts).map((p) => ({ x: p.out / p.conc, y: p.out / run.n * 8 / kw })).sort((a, b) => a.x - b.x);
  if (!kw || !c.length || !(slo > 0)) return null;
  if (slo > c[c.length - 1].x) return { v: 0 };
  if (slo < c[0].x) return { v: c[0].y, ge: true };
  for (let i = 0; i < c.length - 1; i++) {
    if (c[i].x <= slo && slo <= c[i + 1].x) return { v: c[i].y + (c[i + 1].y - c[i].y) * (slo - c[i].x) / ((c[i + 1].x - c[i].x) || 1) };
  }
  return { v: c[c.length - 1].y };
}

// Users at the SLO on an 8-card server (× 8 / devices measured), and on a rack of rackKw filled with
// whole servers (the power left over stays unused). ge: the sweep's largest batch still met the SLO.
const RACK_KW = 15;
function usersPerServer(pts, run, slo) { const u = usersAtSlo(pts, slo); return u ? { v: u.v * 8 / run.n, ge: u.ge } : null; }
function usersPerRack(pts, run, slo, rackKw) {
  const kw = SERVER_KW[family(run.hw)], u = usersPerServer(pts, run, slo);
  if (!kw || !u) return null;
  const servers = Math.floor(rackKw / kw + 1e-9);
  return { v: servers * u.v, ge: u.ge, servers: servers, perServer: u.v, kw: kw };
}

// One number per run and ISL/OSL pair, for the Summary and Releases pages: f(points, run, slo, rackKw).
// bestN: a total scaled to a server, so the best device count of each side is compared. point: a measure
// taken at every batch rather than reduced to one value.
const METRICS = {
  tpkw: { label: 'TPS per kW at SLO', title: 'Token Throughput per Kilowatt', higher: true, bestN: true, slo: true, f: tpsPerKwAtSlo },
  rack: { label: 'Users per rack at SLO', higher: true, bestN: true, slo: true, rack: true, f: usersPerRack },
  users: { label: 'Users at SLO · per 8-card server', higher: true, bestN: true, slo: true, f: usersPerServer },
  // Measures with no one value per run: the pages list the models and draw `point` at every batch.
  // f only picks each side's device count (the best at peak, or at b1).
  tps: { label: 'Output TPS', higher: true, bestN: true, f: (pts) => { const p = peakPoint(pts); return p ? { v: p.out } : null; },
    point: { label: 'Output TPS', f: (p) => p.out } },  // as measured on the run's devices
  total: { label: 'Total TPS', higher: true, bestN: true, f: (pts) => { const p = peakPoint(pts); return p && p.total ? { v: p.total } : null; },
    point: { label: 'Total TPS (input + output)', f: (p) => p.total } },  // as measured
  out: { label: 'Output TPS · per 8-card server', higher: true, bestN: true, f: (pts, r) => { const p = peakPoint(pts); return p ? { v: p.out * 8 / r.n } : null; },
    point: { label: 'Output TPS · per 8-card server', f: (p, r) => p.out * 8 / r.n } },
  tpw: { label: 'Output TPS / W', higher: true, bestN: true, f: (pts) => { const p = peakPoint(pts); return p && p.power > 0 ? { v: p.out / p.power } : null; },
    point: { label: 'Output TPS / W', f: (p) => (p.power > 0 ? p.out / p.power : null) } },
  tpu: { label: 'TPS / User', higher: true, f: (pts) => { const p = c1(pts); return p ? { v: p.out } : null; }, point: { label: 'TPS / User', f: (p) => p.out / p.conc } },
  tpot: { label: 'TPOT p50 (ms)', higher: false, f: (pts) => { const p = c1(pts); return p ? { v: p.tpot50 } : null; }, point: { label: 'TPOT p50 (ms)', f: (p) => p.tpot50 } },
  ttft: { label: 'TTFT p50 (s)', higher: false, f: (pts) => { const p = c1(pts); return p ? { v: p.ttft50 } : null; }, point: { label: 'TTFT p50 (s)', f: (p) => p.ttft50 } },
};
// What an SLO metric was measured against, for titles: '' for the others.
function metricBasis(M, slo, rackKw) { return (M.slo ? ' · SLO ' + slo + ' TPS/user' : '') + (M.rack ? ' · ' + rackKw + ' kW rack' : ''); }

function median(xs) {
  xs = xs.slice().sort((a, b) => a - b);
  if (!xs.length) return null;
  return xs.length % 2 ? xs[(xs.length - 1) / 2] : (xs[xs.length / 2 - 1] + xs[xs.length / 2]) / 2;
}

// ---- formatting ----------------------------------------------------------------------------------
function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000 || Number.isInteger(v)) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
function day(iso) { return iso ? iso.slice(0, 10) : ''; }
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// <select> bound to state[key]; options are strings or {v, label}.
function select(label, key, value, options, attrs) {
  const opts = options.map((o) => {
    const v = typeof o === 'object' ? o.v : o, l = typeof o === 'object' ? o.label : o;
    return '<option value="' + esc(v) + '"' + (String(v) === String(value) ? ' selected' : '') + '>' + esc(l) + '</option>';
  }).join('');
  return '<label class="ctl">' + esc(label) + '<select data-k="' + key + '"' + (attrs || '') + '>' + opts + '</select></label>';
}
function rackInput(kw) {
  return '<label class="ctl">Rack power · kW<input type="number" min="0.1" step="0.1" data-k="rack" data-num value="' + kw + '"></label>';
}
function sloInput(slo) {
  return '<label class="ctl">SLO · tok/s/user<input type="number" min="1" step="1" data-k="slo" data-num value="' + slo + '"></label>';
}
function kpi(label, value, color) {
  return '<div class="kpi"><div class="label">' + esc(label) + '</div><div class="value"' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</div></div>';
}

// ---- router --------------------------------------------------------------------------------------
// The page and its controls live in the URL (#explorer?model=…&scen=…), so a view can be linked.
// Pages register an async render(state) that returns HTML; changing a control re-renders the page.
const PAGES = {};
let STATE = {}, PAGE = 'summary', RENDERING = 0;

function readHash() {
  const [page, query] = location.hash.replace(/^#/, '').split('?');
  PAGE = PAGES[page] ? page : 'summary';
  STATE = Object.fromEntries(new URLSearchParams(query || ''));
  ['slo'].forEach((k) => { if (STATE[k] !== undefined) STATE[k] = +STATE[k]; });
}
function setState(patch) {
  Object.assign(STATE, patch);
  Object.keys(STATE).forEach((k) => { if (STATE[k] === undefined || STATE[k] === null) delete STATE[k]; });
  const q = new URLSearchParams(STATE).toString();
  history.replaceState(null, '', '#' + PAGE + (q ? '?' + q : ''));
  render();
}
async function render() {
  const ticket = ++RENDERING, app = document.getElementById('app');
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + PAGE));
  try {
    const html = await PAGES[PAGE](STATE);
    if (ticket === RENDERING) app.innerHTML = html;  // a newer render started while loading: drop this one
  } catch (e) {
    if (ticket === RENDERING) app.innerHTML = '<div class="status">Could not load the data: ' + esc(e.message) + '</div>';
  }
}
// The measure of a single-point metric (M.point) at every batch, one bar per series in each batch's group,
// with series[ratio[0]]'s multiple over series[ratio[1]] above the group (> 1: better, whichever way the
// measure improves). series: [{ name, color, run }]; failed points are left out.
function batchChart(M, series, ratio, isl, osl) {
  const val = (s, c) => {
    const p = pointsAt(s.run, isl, osl).filter((q) => q.ok && q.conc === c)[0], v = p ? M.point.f(p, s.run) : null;
    return v === null || v === undefined || !isFinite(v) ? null : v;
  };
  const concs = [...new Set(series.flatMap((s) => pointsAt(s.run, isl, osl).map((p) => p.conc)))].sort((a, b) => a - b);
  const rows = concs.map((c) => ({ c: c, v: series.map((s) => val(s, c)) })).filter((x) => x.v.some((v) => v !== null));
  if (!rows.length) return '<div class="empty">No measured batches.</div>';
  const W = 900, H = 230, L = 70, R = 880, T = 30, B = 190, yt = niceTicks(Math.max(...rows.flatMap((x) => x.v.map((v) => v || 0))) * 1.12);
  const y = (v) => B - v / yt.top * (B - T), slot = (R - L) / rows.length, bw = Math.min(26, slot * 0.64 / series.length);
  let s = '';
  yt.list.forEach((v) => { s += '<line x1="' + L + '" x2="' + R + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="var(--grid)"/><text x="' + (L - 8) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + fmt(v) + '</text>'; });
  rows.forEach((x, i) => {
    const cx = L + slot * (i + 0.5), x0 = cx - bw * series.length / 2;
    const a = x.v[ratio[0]], b = x.v[ratio[1]], k = a > 0 && b > 0 ? (M.higher ? a / b : b / a) : null;
    // The batch's values show on hover (data-tip), over a hit area as wide as its slot.
    const tip = 'b' + x.c + '\n' + series.map((sr, j) => sr.name + '  ' + (x.v[j] === null ? '—' : fmt(x.v[j]))).join('\n') + (k === null ? '' : '\n' + k.toFixed(2) + '×');
    s += '<g class="bgrp" data-tip="' + esc(tip) + '"><rect class="hit" x="' + (cx - slot / 2) + '" y="' + T + '" width="' + slot + '" height="' + (B - T) + '" fill="transparent"/>';
    x.v.forEach((v, j) => {
      if (v !== null) s += '<rect x="' + (x0 + bw * j + 1) + '" y="' + y(v) + '" width="' + (bw - 2) + '" height="' + (B - y(v)) + '" fill="' + series[j].color + '"/>';
    });
    s += '</g>';
    const top = Math.min(...x.v.map((v) => (v === null ? B : y(v))));
    if (k !== null) s += '<text class="mult-s" x="' + cx + '" y="' + (top - 8) + '" text-anchor="middle" style="fill:' + (k > 1.02 ? 'var(--accent)' : k < 0.98 ? 'var(--muted)' : 'var(--text)') + '">' + k.toFixed(2) + '×</text>';  // ±2%: no change
    s += '<text x="' + cx + '" y="' + (B + 18) + '" text-anchor="middle">b' + x.c + '</text>';
  });
  s += '<line x1="' + L + '" x2="' + R + '" y1="' + B + '" y2="' + B + '" stroke="var(--faint)"/>'
    + '<text class="axis-title" transform="translate(14 ' + (T + B) / 2 + ') rotate(-90)" text-anchor="middle">' + esc(M.point.label.toUpperCase()) + '</text>'
    + '<text class="axis-title" x="' + (L + R) / 2 + '" y="' + (H - 4) + '" text-anchor="middle">BATCH (CONCURRENCY)</text>';
  const legend = '<div class="legend">' + series.map((x) => '<span><span class="swatch" style="background:' + x.color + '"></span>' + esc(x.name) + '</span>').join('')
    + (series.length > 1 ? '<span class="faint">× = ' + esc(series[ratio[0]].name) + ' vs ' + esc(series[ratio[1]].name) + '</span>' : '') + '</div>';
  return legend + '<svg class="chart" viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="' + esc(M.point.label) + ' by batch">' + s + '</svg>';
}

// [data-tip]: its text in a box that follows the pointer.
function showTip(e) {
  let box = document.getElementById('tip');
  const el = e.target.closest && e.target.closest('[data-tip]');
  if (!el) { if (box) box.hidden = true; return; }
  if (!box) { box = document.createElement('div'); box.id = 'tip'; box.className = 'tip'; document.body.appendChild(box); }
  box.textContent = el.dataset.tip;
  box.hidden = false;
  const left = e.clientX + 16 + box.offsetWidth > innerWidth ? e.clientX - 16 - box.offsetWidth : e.clientX + 16;
  box.style.left = left + scrollX + 'px';
  box.style.top = e.clientY + scrollY - box.offsetHeight / 2 + 'px';
}

// One-of-many picker drawn as tiles (radio buttons bound to state[key], so the pick is kept in the link).
// items: [{ key, name, sub }]; returns the picked key (state[key] if it is one of the items, else dflt).
function pickedKey(st, key, items, dflt) { return items.some((x) => x.key === st[key]) ? st[key] : dflt; }
function pickTiles(key, items, pick) {
  return '<div class="tiles">' + items.map((x) => {
    const on = x.key === pick;
    return '<label class="tile' + (on ? ' on' : '') + '"><input type="radio" name="' + key + '" data-k="' + key + '" value="' + esc(x.key) + '"' + (on ? ' checked' : '') + '>'
      + '<span class="name">' + esc(x.name) + '</span><span class="sub">' + esc(x.sub || '') + '</span></label>';
  }).join('') + '</div>';
}

function start() {
  const app = document.getElementById('app');
  app.addEventListener('change', (e) => {
    const el = e.target;
    if (el.dataset.k) setState({ [el.dataset.k]: el.dataset.num !== undefined ? (+el.value > 0 ? +el.value : undefined) : el.value });
    else if (el.dataset.toggle !== undefined) PAGES[PAGE].toggle(el.dataset.toggle, el.checked);
  });
  document.addEventListener('pointermove', showTip);
  window.addEventListener('hashchange', () => { readHash(); render(); });
  readHash();
  render();
}
