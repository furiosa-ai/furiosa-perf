// Data: every run in the site, filterable, with its points as JSON, the filtered list as CSV, and one CSV of
// every point of the filtered runs per platform (device + serving framework version) and per model.
'use strict';

const RUN_COLUMNS = ['kind', 'task', 'model_id', 'hw', 'n', 'layout', 'backend', 'ver', 'date', 'num_points', 'failed', 'flags', 'variant', 'id'];

function filteredRuns(runs, st) {
  return runs.filter((r) => (!st.kind || r.kind === st.kind) && (!st.dtask || r.task === st.dtask) && (!st.hw || r.hw === st.hw)
    && (!st.q || (r.model_id + ' ' + r.ver).toLowerCase().indexOf(st.q.toLowerCase()) >= 0));
}

function csvCell(v) { v = Array.isArray(v) ? v.join(' ') : v === null || v === undefined ? '' : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function csvOf(runs) {
  return [RUN_COLUMNS.join(',')].concat(runs.map((r) => RUN_COLUMNS.map((c) => csvCell(r[c])).join(','))).join('\n') + '\n';
}

// Every point of the given runs (one model's, or one platform's), one row each: which run (POINT_RUN_COLUMNS), then every recorded value.
// Runs written by different versions record different columns: the union, left empty where a run has none.
const POINT_RUN_COLUMNS = ['model_id', 'task', 'kind', 'hw', 'n', 'layout', 'backend', 'ver', 'date', 'variant', 'flags', 'id'];
async function modelCsv(runs) {
  await loadPoints(runs);
  const cols = [];
  runs.forEach((r) => DATA.points[r.id].forEach((p) => Object.keys(p.raw).forEach((c) => { if (cols.indexOf(c) < 0) cols.push(c); })));
  const head = POINT_RUN_COLUMNS.map((c) => (c === 'id' ? 'run_id' : c)).concat(cols);
  const lines = runs.flatMap((r) => DATA.points[r.id].map((p) => POINT_RUN_COLUMNS.map((c) => csvCell(r[c])).concat(cols.map((c) => csvCell(p.raw[c]))).join(',')));
  return [head.join(',')].concat(lines).join('\n') + '\n';
}
function save(text, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

PAGES.data = async function (st) {
  const runs = await loadIndex();
  const uniq = (key) => [...new Set(runs.map((r) => r[key]))].sort();
  const rows = filteredRuns(runs, st).sort((a, b) => famOrder(a.hw) - famOrder(b.hw) || a.hw.localeCompare(b.hw) || a.n - b.n
    || cmpVer(b.ver, a.ver) || a.task.localeCompare(b.task) || a.model.localeCompare(b.model));
  const points = rows.reduce((s, r) => s + r.num_points, 0), failed = rows.reduce((s, r) => s + r.failed, 0);
  const any = (label) => [{ v: '', label: label }];
  const tile = (attr, key, name, rs) => '<button class="tile" ' + attr + '="' + esc(key) + '"><span class="name">' + esc(name) + '</span><span class="sub">'
    + rs.length + ' runs · ' + rs.reduce((s, r) => s + r.num_points, 0).toLocaleString('en-US') + ' points · CSV</span></button>';
  // Newest version first within each device.
  const plats = [...new Set(rows.map(platformKey))].map((k) => rows.filter((r) => platformKey(r) === k)[0])
    .sort((a, b) => famOrder(a.hw) - famOrder(b.hw) || a.hw.localeCompare(b.hw) || cmpVer(b.ver, a.ver)).map(platformKey).map((k) => tile('data-platform-csv', k, platformLabel(k), rows.filter((r) => platformKey(r) === k))).join('');
  const models = [...new Set(rows.map((r) => r.model))].sort((a, b) => a.localeCompare(b)).map((m) => {
    return tile('data-model-csv', m, m, rows.filter((r) => r.model === m));
  }).join('');
  const table = rows.map((r) => '<tr><td>' + esc(r.kind) + '</td><td>' + esc(r.task) + '</td><td>' + esc(r.model_id) + '</td>'
    + '<td>' + esc(r.hw + ' ×' + r.n + (r.layout ? ' ' + r.layout : '')) + '</td><td>' + esc(backendLabel(r.backend) + ' ' + r.ver) + '</td>'
    + '<td class="date">' + day(r.date) + '</td><td class="num">' + r.num_points + '</td>'
    + '<td class="num" style="color:' + (r.failed ? 'var(--fail)' : 'var(--faint)') + '">' + r.failed + '</td>'
    + '<td class="muted">' + esc((r.flags || []).join(' ')) + '</td><td>' + esc(r.variant || '') + '</td>'
    + '<td><a href="#run?id=' + encodeURIComponent(r.id) + '">Details</a> · <a href="data/runs/' + encodeURIComponent(r.id) + '.json" download>JSON</a></td></tr>').join('');
  return '<div class="head"><h1>Data</h1><div class="controls">'
    + select('Kind', 'kind', st.kind || '', any('All').concat(uniq('kind')))
    + select('Task', 'dtask', st.dtask || '', any('All').concat(uniq('task')))
    + select('Hardware', 'hw', st.hw || '', any('All').concat(uniq('hw')))
    + '<label class="ctl">Model or version<input type="search" data-k="q" value="' + esc(st.q || '') + '" placeholder="e.g. Qwen3 or 2026.4"></label>'
    + '<button class="button" id="csv">Download CSV</button></div></div>'
    + '<section class="kpis">' + kpi('Runs', rows.length.toLocaleString('en-US')) + kpi('Points', points.toLocaleString('en-US'))
    + kpi('Models', new Set(rows.map((r) => r.model)).size) + kpi('Failed points', failed.toLocaleString('en-US'), 'var(--fail)') + '</section>'
    + (plats ? '<section class="panel"><div class="panel-head"><h2>Download by platform</h2><span class="muted">One CSV per device and serving framework version: every point of its runs shown below (all models and device counts), the run it came from first</span></div>'
      + '<div class="tiles">' + plats + '</div></section>' : '')
    + (models ? '<section class="panel"><div class="panel-head"><h2>Download by model</h2><span class="muted">One CSV per model: every point of its runs shown below, the run it came from first</span></div>'
      + '<div class="tiles">' + models + '</div></section>' : '')
    + '<section class="panel"><div class="scroll"><table class="list" style="min-width:1000px"><thead><tr><th>Kind</th><th>Task</th><th>Model</th><th>Hardware</th><th>Backend</th>'
    + '<th>Measured</th><th class="num">Points</th><th class="num">Failed</th><th>Server flags</th><th>Variant</th><th></th></tr></thead><tbody>' + table + '</tbody></table></div>'
    + '<p class="caption">A failed point is one the run judged invalid (client error, failure rate over 5%, no output) or, for results from before the presets, one with no output. '
    + 'JSON has every recorded value of a run\'s points; the CSV lists the runs shown here.</p></section>';
};

// The CSVs are built from the filters in the URL when a button is clicked.
document.addEventListener('click', async (e) => {
  const tile = e.target.closest('[data-model-csv]'), plat = e.target.closest('[data-platform-csv]');
  if (e.target.id === 'csv') save(csvOf(filteredRuns(await loadIndex(), STATE)), 'furiosa-perf-runs.csv');
  if (tile) {
    const m = tile.dataset.modelCsv, runs = filteredRuns(await loadIndex(), STATE).filter((r) => r.model === m);
    save(await modelCsv(runs), 'furiosa-perf_' + m.replace(/[^\w.-]+/g, '-') + '.csv');
  }
  if (plat) {
    const k = plat.dataset.platformCsv, runs = filteredRuns(await loadIndex(), STATE).filter((r) => platformKey(r) === k);
    save(await modelCsv(runs), 'furiosa-perf_' + k.replace(/[^\w.-]+/g, '_') + '.csv');
  }
});
