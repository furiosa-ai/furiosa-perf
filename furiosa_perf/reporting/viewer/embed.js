// Embedding & Rerank (Explorer, tasks embedding and rerank): one model's runs over every input length and concurrency the presets cover,
// one curve per run (device × count × backend version). No output tokens here: throughput is input tokens,
// texts (embedding) or query-document pairs (rerank) per second, and the latency is the whole request (E2EL).
'use strict';

const EMBED_TASKS = { embedding: 'Embedding', rerank: 'Rerank' };
// What a request carries: embedding sends `batch` texts (1 = a query, more = an indexing batch of chunks),
// rerank sends one query with `batch` documents. The embedding query rows run open loop (one request per user
// per second, see the preset), everything else closed loop.
function batchLabel(task, b) {
  if (task === 'rerank') return b + ' documents / query';
  return b === 1 ? '1 text / request · queries, RPS = concurrency' : b + ' texts / request · indexing';
}
function tokensPerSec(p) { return p.total > 0 ? p.total : p.seq > 0 ? p.seq * p.isl : null; }  // rerank reports no usage: pairs × pair length
const EMBED_Y = {
  tok: { label: (t) => 'Total TPS', f: tokensPerSec, higher: true },
  seq: { label: (t) => 'Pairs / s', f: (p) => p.seq, higher: true, only: 'rerank' },  // documents scored per second
  reqps: { label: (t) => 'RPS', f: (p) => p.reqps, higher: true },
  e2el50: { label: (t) => 'E2EL p50 (s)', f: (p) => p.e2el50, higher: false },
  e2el99: { label: (t) => 'E2EL p99 (s)', f: (p) => p.e2el99, higher: false },
  duration: { label: (t) => 'Run duration (s)', f: (p) => p.duration, higher: false },
};
const EMBED_X = { conc: 'Concurrency' };
const EW = 900, EH = 460, EL = 70, ER = 880, ET = 20, EB = 410;

// Runs of the embedding and rerank presets as configured, newest of each measurement.
async function embedRuns() {
  return latestRuns((await loadIndex()).filter((r) => r.kind === 'baseline' && !r.variant && EMBED_TASKS[r.task]));
}
const log2 = (v) => Math.log(v) / Math.LN2;
function fmtLen(v) { return v >= 1024 && v % 1024 === 0 ? v / 1024 + 'K' : String(v); }

// What each axis measures, in plain words, for the note under the chart.
const EMBED_X_TEXT = {
  conc: 'requests in flight at once; for the query rows it is also the RPS sent (one request per user per second).',
};
function embedYText(key, task) {
  const unit = task === 'rerank' ? 'query + document pair' : 'text';
  return {
    tok: 'input tokens the server ' + (task === 'rerank' ? 'scored' : 'embedded') + ' per second (there are no output tokens)' + (task === 'rerank' ? '; the server reports no token usage for /v1/rerank, so this is pairs/s × pair length' : '') + '.',
    seq: 'query-document pairs scored per second.',
    reqps: 'requests completed per second; a request is one ' + (task === 'rerank' ? 'query with its documents' : 'text, or a batch of texts for indexing') + '.',
    e2el50: 'median time from sending a request to its full response; for the query rows it includes any wait in the server.',
    e2el99: '99th-percentile time from sending a request to its full response; for the query rows it includes any wait in the server.',
    duration: 'how long the point ran; the query rows send requests for 30 s, so a longer run means the server fell behind the arrival rate.',
  }[key] || '';
}

async function renderEmbed(st, task, asideTop) {
  const runs = await embedRuns();
  const tRuns = runs.filter((r) => r.task === task);
  if (!tRuns.length) return '<div class="status">No ' + task + ' results.</div>';
  const models = [...new Set(tRuns.map((r) => r.model))].sort((a, b) => a.localeCompare(b));
  const model = models.indexOf(st.model) >= 0 ? st.model : models[0];
  const mRuns = tRuns.filter((r) => r.model === model).sort((a, b) => famOrder(a.hw) - famOrder(b.hw) || a.n - b.n || cmpVer(b.ver, a.ver));
  await loadPoints(mRuns);
  const ptsOf = (r) => DATA.points[r.id] || [];
  const batches = [...new Set(mRuns.flatMap((r) => ptsOf(r).map((p) => p.batch)))].sort((a, b) => a - b);
  const batch = batches.indexOf(+st.batch) >= 0 ? +st.batch : batches[0];
  const sel = (r) => ptsOf(r).filter((p) => p.batch === batch);
  const lens = [...new Set(mRuns.flatMap((r) => sel(r).map((p) => p.isl)))].sort((a, b) => a - b);
  const concs = [...new Set(mRuns.flatMap((r) => sel(r).map((p) => p.conc)))].sort((a, b) => a - b);
  const xKey = 'conc';
  // The other dimension is held at one value: a concurrency when sweeping length, a length when sweeping concurrency.
  const len = lens.indexOf(+st.len) >= 0 ? +st.len : lens.indexOf(512) >= 0 ? 512 : lens[0];
  const yKeys = Object.keys(EMBED_Y).filter((k) => !EMBED_Y[k].only || EMBED_Y[k].only === task);
  const yKey = yKeys.indexOf(st.y) >= 0 ? st.y : (task === 'rerank' ? 'seq' : 'tok');
  const Y = EMBED_Y[yKey];
  const slo = st.eslo > 0 ? +st.eslo : 0;  // E2EL limit in s, drawn when a latency is on Y

  // Series: colour by hardware family (as the Explorer), the newest version strong.
  const newest = newestVersions(mRuns);
  const uniq = (xs) => xs.filter((x, i) => xs.indexOf(x) === i);
  const all = mRuns.map((r) => {
    const same = mRuns.filter((x) => x.hw === r.hw), vers = uniq(same.map((x) => x.ver)).sort((a, b) => cmpVer(b, a)), ns = uniq(same.map((x) => x.n)).sort((a, b) => a - b);
    const rngd = family(r.hw) === 'rngd', rank = ns.indexOf(r.n), shade = rngd ? Math.max(0, rank - 1) : rank;
    const base = 'var(--c-' + (rngd && rank ? 'orange' : hueOf(r.hw)) + '-' + Math.min(3, vers.indexOf(r.ver)) + ')';
    return { r: r, label: seriesLabel(r), dash: ['none', '6 4', '2 4', '10 4 2 4'][Math.min(3, vers.indexOf(r.ver))],
      color: shade ? 'color-mix(in srgb, ' + base + ' ' + Math.max(40, 100 - 25 * shade) + '%, var(--panel))' : base };
  });
  const on = st.on && all.some((o) => st.on.split('~').indexOf(o.r.id) >= 0) ? st.on.split('~') : null;
  all.forEach((o) => { o.on = on ? on.indexOf(o.r.id) >= 0 : o.r.ver === newest[o.r.hw]; });
  SHOWN = all.filter((o) => o.on).map((o) => o.r.id);
  const visible = all.filter((o) => o.on);

  // The curve of each series: Y against length at the chosen concurrency, or against concurrency at the chosen length.
  const xs = concs;
  const at = (o, x) => sel(o.r).filter((p) => p.conc === x && p.isl === len)[0];
  let ymax = 0;
  const plotted = visible.map((o) => {
    const vals = [];
    xs.forEach((x) => {
      const p = at(o, x), y = p && p.ok ? Y.f(p) : null;
      if (y === null || y === undefined || !isFinite(y)) return;
      ymax = Math.max(ymax, y); vals.push({ p: p, x: x, y: y });
    });
    return { o: o, vals: vals };
  });
  if (slo && !Y.higher) ymax = Math.max(ymax, slo);
  const yt = niceTicks(ymax * 1.05);
  // X is log2: the presets double the length and the concurrency at every step.
  const x0 = log2(xs[0] || 1), x1 = log2(xs[xs.length - 1] || 2), span = x1 - x0 || 1;
  const px = (x) => EL + (log2(x) - x0) / span * (ER - EL), py = (y) => EB - y / yt.top * (EB - ET);
  let svg = '<line x1="' + EL + '" y1="' + EB + '" x2="' + ER + '" y2="' + EB + '" stroke="var(--faint)"/><line x1="' + EL + '" y1="' + ET + '" x2="' + EL + '" y2="' + EB + '" stroke="var(--faint)"/>';
  xs.forEach((x) => { svg += '<line x1="' + px(x) + '" x2="' + px(x) + '" y1="' + ET + '" y2="' + EB + '" stroke="var(--grid)"/><text x="' + px(x) + '" y="' + (EB + 20) + '" text-anchor="middle">b' + x + '</text>'; });
  yt.list.forEach((v) => { svg += '<text x="' + (EL - 8) + '" y="' + (py(v) + 4) + '" text-anchor="end">' + fmt(v) + '</text>'; });
  svg += '<text class="axis-title" x="' + (EL + ER) / 2 + '" y="' + (EH - 4) + '" text-anchor="middle">' + esc(EMBED_X[xKey].toUpperCase()) + '</text>';
  svg += '<text class="axis-title" transform="translate(14 ' + (ET + EB) / 2 + ') rotate(-90)" text-anchor="middle">' + esc(Y.label(task).toUpperCase()) + '</text>';
  if (slo && !Y.higher) svg += '<line x1="' + EL + '" x2="' + ER + '" y1="' + py(slo) + '" y2="' + py(slo) + '" stroke="var(--faint)" stroke-dasharray="2 4"/><text x="' + (ER - 4) + '" y="' + (py(slo) - 6) + '" text-anchor="end">E2EL limit ' + slo + ' s</text>';
  // The server counted fewer tokens per text than the ISL: it truncated the input to its limit (bge-m3 past 8192).
  const cut = (p) => (p.tokText > 0 && p.tokText < p.isl * 0.9 ? Math.round(p.tokText) : 0);
  const tipOf = (p) => 'ISL ' + p.isl + ' · b' + p.conc + ' · ' + batchLabel(task, p.batch).split(' · ')[0]
    + '\nTotal TPS ' + fmt(tokensPerSec(p)) + '\n' + Y.label(task).replace(/ \(s\)$/, '') + ' ' + fmt(Y.f(p))
    + '\nRPS ' + fmt(p.reqps) + (task === 'embedding' && p.batch === 1 ? ' (sent ' + p.conc + ')' : '') + '\nE2EL p50 ' + fmt(p.e2el50) + ' s · p99 ' + fmt(p.e2el99) + ' s\nDuration ' + fmt(p.duration) + ' s' + (cut(p) ? '\nTruncated by the server to ~' + cut(p) + ' tokens per text' : '');
  plotted.forEach((g) => {
    if (g.vals.length > 1) svg += '<path d="' + smoothPath(g.vals.map((v) => ({ x: px(v.x), y: py(v.y) }))) + '" fill="none" stroke="' + g.o.color + '" stroke-width="2.5" stroke-dasharray="' + g.o.dash + '"/>';
    g.vals.forEach((v) => { svg += '<circle cx="' + px(v.x) + '" cy="' + py(v.y) + '" r="3.5" fill="' + g.o.color + '" stroke="var(--panel)" stroke-width="1.5"><title>' + esc(g.o.label + '\n' + tipOf(v.p)) + '</title></circle>'; });
  });
  HOVER = null;  // no pointer readout on a log axis: the points carry their values

  // Every point of the chosen series as a length × concurrency grid, so the whole range the presets cover reads at once.
  const tItems = visible.map((o) => ({ key: o.r.id, name: hwName(o.r.hw) + ' ×' + o.r.n, sub: backendLabel(o.r.backend) + ' ' + o.r.ver }));
  const tKey = pickedKey(st, 'table', tItems, tItems.length ? tItems[0].key : null), tSeries = visible.filter((o) => o.r.id === tKey)[0];
  // Second series for the ratio (RNGD over the other, oriented so > 1 is better): the first visible one of another family.
  const vsSeries = tSeries && visible.filter((o) => o.r.id !== tKey && family(o.r.hw) !== family(tSeries.r.hw))[0];
  const cellOf = (o, L, c) => sel(o.r).filter((p) => p.isl === L && p.conc === c)[0];
  const grid = !tSeries ? '' : '<div class="scroll"><table class="list emb-grid"><thead><tr><th class="num">ISL \\ batch</th>' + concs.map((c) => '<th class="num">b' + c + '</th>').join('') + '</tr></thead><tbody>'
    + lens.map((L) => '<tr><td class="num">' + fmtLen(L) + '</td>' + concs.map((c) => {
      const p = cellOf(tSeries, L, c);
      if (!p) return '<td class="num faint">·</td>';
      const v = Y.f(p), q = vsSeries && cellOf(vsSeries, L, c), w = q && q.ok ? Y.f(q) : null;
      const k = p.ok && v > 0 && w > 0 ? (Y.higher ? v / w : w / v) : null;
      const low = task === 'embedding' && p.batch === 1 && p.reqps > 0 && p.reqps < p.conc * 0.9;  // open loop: fewer requests/s than the rate asked = overloaded
      return '<td class="num' + (p.ok ? '' : ' faint') + '" title="' + esc(tipOf(p) + (q ? '\n\n' + vsSeries.label + '\n' + tipOf(q) : '')) + '">' + (p.ok ? fmt(v) : '<span class="fail">' + esc(p.raw.fail_reason || 'failed') + '</span>')
        + (low ? '<span class="sub muted" title="RPS achieved, below the RPS sent">' + fmt(p.reqps) + ' RPS</span>' : '')
        + (cut(p) ? '<span class="sub muted" title="The server truncated every text to this many tokens">↓' + cut(p) + ' tok</span>' : '')
        + (k !== null ? '<span class="sub" style="color:' + (k >= 1 ? 'var(--accent)' : 'var(--muted)') + '">' + multText(k, Y.higher) + '</span>' : '') + '</td>';
    }).join('') + '</tr>').join('') + '</tbody></table></div>';

  const seriesList = all.map((o) => '<label class="series"><input type="checkbox" data-toggle="' + esc(o.r.id) + '"' + (o.on ? ' checked' : '') + '>'
    + '<svg width="28" height="10" aria-hidden="true"><line x1="0" y1="5" x2="28" y2="5" stroke="' + o.color + '" stroke-width="2" stroke-dasharray="' + o.dash + '"/></svg><span>' + esc(o.label) + '</span></label>').join('');
  const legend = '<div class="legend chart-legend">' + visible.map((o) => '<a class="run-link" title="How this run was measured" href="#run?id=' + encodeURIComponent(o.r.id) + '"><svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="' + o.color
    + '" stroke-width="3" stroke-linecap="round" stroke-dasharray="' + o.dash + '"/></svg><b>' + esc(hwName(o.r.hw) + ' ×' + o.r.n) + '</b><span class="mono">' + esc(backendLabel(o.r.backend) + ' ' + o.r.ver) + '</span><span class="faint">↗</span></a>').join('') + '</div>';
  const held = 'ISL ' + len;
  const subtitle = model + ' · ' + batchLabel(task, batch) + ' · ' + held;
  EXPORTS.main = { w: EW, h: EH, title: Y.label(task), subtitle: subtitle, file: (model + '_' + task + '_b' + batch + '_' + yKey + '_vs_' + xKey).replace(/[^\w.-]+/g, '-'),
    legend: visible.map((o) => ({ color: o.color, dash: o.dash, name: hwName(o.r.hw) + ' ×' + o.r.n, ver: backendLabel(o.r.backend) + ' ' + o.r.ver })) };
  const how = task === 'rerank'
    ? 'Every request is one query with ' + batch + ' documents, each query-document pair ISL tokens long; the client sends the next request as soon as one returns (closed loop), ' + 'concurrency' + ' queries at once. Pairs/s = documents scored per second. The server reports no token usage for /v1/rerank, so input tokens/s = pairs/s × pair length.'
    : batch === 1
      ? 'Queries: one text per request. Every user sends one request a second (RPS sent = concurrency), so E2EL includes any wait in the server; when a server cannot keep up, the RPS achieved falls below the RPS sent (marked in the grid) and the run takes longer than the arrival window.'
      : 'Indexing: ' + batch + ' chunks of ISL tokens per request, sent closed loop (the next batch as soon as one returns) at the given concurrency; input tokens/s is the indexing speed.';
  return '<div class="explorer"><aside class="panel">'
    + asideTop + select('Model', 'model', model, models)
    + select('Request', 'batch', String(batch), batches.map((b) => ({ v: String(b), label: batchLabel(task, b) })))
    + select('ISL', 'len', String(len), lens.map((L) => ({ v: String(L), label: fmtLen(L) + ' tokens' })))
    + '<div class="ctl">Hardware · backend' + seriesList + '</div></aside><section>'
    + '<div class="controls">' + select('X axis', 'x', xKey, [{ v: 'conc', label: 'Concurrency' }])
    + select('Y axis', 'y', yKey, yKeys.map((k) => ({ v: k, label: EMBED_Y[k].label(task) })))
    + (!Y.higher ? '<label class="ctl">E2EL limit · s<input type="number" min="0" step="0.1" data-k="eslo" data-num value="' + (slo || '') + '" placeholder="none"></label>' : '') + '</div>'
    + '<div class="panel"><div class="panel-head"><div><h2 style="font-weight:400;font-size:18px">' + esc(Y.label(task)) + '</h2><div class="muted">' + esc(subtitle) + '</div></div>' + downloads('main') + '</div>' + legend
    + '<div class="chart-wrap"><svg class="chart" data-chart="main" viewBox="0 0 ' + EW + ' ' + EH + '" width="100%" role="img" aria-label="' + esc(Y.label(task) + ' against ' + EMBED_X[xKey] + ' for ' + model) + '">' + svg + '<g class="hover"></g></svg><div class="hover-tip" hidden></div></div>'
    + '<div class="axis-note"><div><b>X · ' + esc(EMBED_X[xKey]) + '</b> — ' + esc(EMBED_X_TEXT[xKey]) + '</div><div><b>Y · ' + esc(Y.label(task)) + '</b> — ' + esc(embedYText(yKey, task)) + '</div>'
    + '<div><b>Request · ' + esc(batchLabel(task, batch)) + '</b> — ' + esc(how) + '</div></div>'
    + note('embed', (visible.length ? visible.map((o) => o.label).join(', ') + '. ' : '') + 'Failed points are left off the curve (greyed in the grid).', '') + '</div>'
    + '<div class="panel batches"><div class="panel-head"><h2 style="font-weight:400;font-size:18px">Every length × concurrency</h2><span class="muted">' + esc(Y.label(task) + ' · ' + batchLabel(task, batch)) + (vsSeries ? ' · × = ' + esc(hwName(tSeries.r.hw) + ' ×' + tSeries.r.n + ' vs ' + hwName(vsSeries.r.hw) + ' ×' + vsSeries.r.n) : '') + '</span></div>'
    + (tItems.length > 1 ? pickTiles('table', tItems, tKey) : '') + (grid || '<div class="empty">No series shown.</div>') + '</div></section></div>';
};
// Old links: #embed opens the Explorer on the embedding task.
PAGES.embed = (st) => PAGES.explorer(Object.assign({}, st, { task: EMBED_TASKS[st.task] ? st.task : 'embedding' }));
PAGES.embed.toggle = (id, checked) => PAGES.explorer.toggle(id, checked);
