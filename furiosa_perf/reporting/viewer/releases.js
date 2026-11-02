// Releases: one RNGD metric per model across furiosa-llm releases, tinted by the change from the previous one.
'use strict';

const FLAG_LABEL = { 'enable-overlap-scheduling': 'overlap', 'experimental-enable-npu-sampling': 'npu-sampling' };

// Cell tint from a ratio (> 1 = better): improvements in the accent, regressions in grey;
// full strength at 1.2× / 0.8×, none within 0.98–1.02×.
function tint(r) {
  if (r === null || r === undefined || Math.abs(r - 1) <= 0.02) return 'transparent';
  const a = Math.min(Math.abs(r - 1), 0.2) / 0.2;
  return r > 1 ? 'rgba(226,21,0,' + (0.12 + a * 0.73).toFixed(2) + ')' : 'rgba(169,169,169,' + (0.06 + a * 0.39).toFixed(2) + ')';
}
function times(v, higher) { return v === null || v === undefined ? '—' : multText(v, higher !== false); }

PAGES.releases = async function (st) {
  const gen = await generationRuns();
  const tasks = Object.keys(GEN_TASKS).filter((t) => gen.some((r) => r.task === t && family(r.hw) === 'rngd'));
  const task = tasks.indexOf(st.task) >= 0 ? st.task : tasks[0];
  const all = gen.filter((r) => family(r.hw) === 'rngd' && r.task === task);
  const devs = [...new Set(all.map((r) => r.n))].sort((a, b) => a - b);
  const dev = st.dev && devs.indexOf(+st.dev) >= 0 ? st.dev : 'all';
  const sel = all.filter((r) => dev === 'all' || r.n === +dev);
  const scens = shapeOptions(sel);
  const scen = pickShape(st.scen, scens);
  const metric = METRICS[st.metric] ? st.metric : 'tpkw';
  const M = METRICS[metric];
  const slo = st.slo > 0 ? st.slo : 20, rackKw = +st.rack > 0 ? +st.rack : RACK_KW;
  const show = showOf(st);
  const controls = sideGroup('RNGD', select('Devices', 'dev', dev, [{ v: 'all', label: 'All device counts' }].concat(devs.map((n) => ({ v: String(n), label: 'RNGD ×' + n })))))
    + sideGroup('Scenario', select('Task', 'task', task, tasks.map((t) => ({ v: t, label: GEN_TASKS[t] }))) + select('ISL / OSL', 'scen', scen, scens)
      + (M.slo ? sloInput(slo) : '') + (M.rack ? rackInput(rackKw) : '') + (M.kw ? serverKwInputs(['rngd']) : ''))
    + sideGroup('View', select('Metric', 'metric', metric, Object.keys(METRICS).map((k) => ({ v: k, label: METRICS[k].label }))) + showSelect(show));
  const page = (body) => sidePage('Releases', controls, body);
  if (!scen) return page('<div class="status">No RNGD results.</div>');

  await loadPoints(sel);
  const [isl, osl] = scen.split('/').map(Number);
  // One row per model × device count, one column per release.
  const vers = [], byRow = {}, runsOf = {}, flags = {}, dates = {};
  sel.forEach((r) => {
    const pts = pointsAt(r, isl, osl), m = pts.length ? M.f(pts, r, slo, rackKw) : null;
    if (!m || !(m.v > 0)) return;
    if (vers.indexOf(r.ver) < 0) vers.push(r.ver);
    const k = r.model + '|' + r.n;
    (byRow[k] = byRow[k] || {})[r.ver] = m;
    (runsOf[k] = runsOf[k] || {})[r.ver] = r;
    flags[r.ver] = flags[r.ver] || {};
    (r.flags || []).forEach((f) => { flags[r.ver][f] = 1; });
    if (r.date && (!dates[r.ver] || r.date < dates[r.ver])) dates[r.ver] = r.date;
  });
  vers.sort(cmpVer);
  // Ratio of the newer release to the older, oriented so that > 1 is an improvement.
  const better = (a, b) => (M.higher ? b.v / a.v : a.v / b.v);
  const rows = [];
  Object.keys(byRow).forEach((k) => {
    const vals = byRow[k], steps = {};
    let prev = null, first = null;
    vers.forEach((v) => {
      if (!vals[v]) return;
      if (prev) steps[v] = better(vals[prev], vals[v]);
      if (!first) first = v;
      prev = v;
    });
    if (first === prev) return;  // measured on one release only
    const [model, n] = k.split('|');
    rows.push({ key: k, model: model, n: +n, vals: vals, steps: steps, total: better(vals[first], vals[prev]), first: first, last: prev });
  });
  rows.sort((a, b) => b.total - a.total || a.model.localeCompare(b.model));

  const latest = vers[vers.length - 1];
  const total = rows.length;
  if (show !== 'all') rows.splice(show);
  const more = moreButton(rows.length, total);
  const short = (v) => v.replace(/^\d{4}\./, '');

  const cell = (main, o) => {
    o = o || {};
    return '<div class="' + (o.cls || '') + '" style="background:' + (o.bg || 'transparent') + (o.color ? ';color:' + o.color : '') + (o.weight ? ';font-weight:' + o.weight : '') + '">'
      + '<span>' + main + '</span>' + (o.sub ? '<span class="sub" style="color:' + (o.subColor || 'var(--muted)') + '">' + o.sub + '</span>' : '') + '</div>';
  };
  let cells = cell('Model', { cls: 'hdr left' }) + cell('RNGD', { cls: 'hdr' });
  vers.forEach((v, j) => {
    // Release, date and the server flags it turned on (new ones in white).
    const fl = Object.keys(flags[v] || {}).sort(), pf = j ? flags[vers[j - 1]] || {} : {};
    const added = fl.filter((f) => !pf[f]).map((f) => '+' + (FLAG_LABEL[f] || f));
    cells += cell(esc(short(v)), { cls: 'hdr', weight: 600, color: v === latest ? 'var(--title)' : 'var(--muted)',
      sub: esc(day(dates[v]).slice(0, 7) + (added.length ? '  ' + added.join(' ') : '')), subColor: added.length ? 'var(--text)' : 'var(--faint)' });
  });
  cells += cell('Total', { cls: 'hdr' });
  rows.forEach((r) => {
    cells += cell(esc(r.model), { cls: 'left' }) + cell('×' + r.n, { color: 'var(--muted)' });
    vers.forEach((v) => {
      const x = r.vals[v], b = r.steps[v];
      cells += x ? cell((x.ge ? '≥' : '') + fmt(x.v), { bg: tint(b), sub: b === undefined ? '' : times(b, M.higher), subColor: b > 1.02 ? 'var(--title)' : 'var(--muted)' })
        : cell('—', { color: 'var(--faint)' });
    });
    cells += cell(times(r.total, M.higher), { weight: 600, color: r.total > 1.02 ? 'var(--accent)' : r.total < 0.98 ? 'var(--muted)' : 'var(--text)',
      sub: esc(short(r.first) + ' → ' + short(r.last)), subColor: 'var(--faint)' });
  });

  // Panel heading: the metric in plain words, one line of conditions, the method folded (as on RNGD vs GPU).
  const HEAD = metricText(metric, M, slo, rackKw);
  const cond = GEN_TASKS[task] + ', ' + scen + ' ISL/OSL. ' + (dev === 'all' ? 'Every RNGD device count.' : 'RNGD ×' + dev + '.')
    + (M.kw ? ' 8-card server power: RNGD ' + serverKw('RNGD') + ' kW.' : '') + (M.slo ? ' ≥ = lower bound.' : '');
  const how = HEAD[1] + (M.point ? ' Pick a model to see every batch, one bar per release.'
    : ' Each cell is the value on that release; the small figure is the multiple over the previous release that measured the model (' + (M.higher ? 'higher' : 'lower')
      + ' is better). Total = first measured → latest. The header marks server flags a release turned on (e.g. +overlap = --enable-overlap-scheduling).')
    + (M.slo ? ' ≥: still above ' + slo + ' tok/s per user at the largest batch measured.' : '');
  const head = (extra) => '<div class="panel-head"><div><h2>' + esc(HEAD[0] + ' by furiosa-llm release') + '</h2><div class="panel-sub">' + note('releases', cond, how) + '</div></div>' + (extra || '') + '</div>';
  if (M.point) {
    // No one value per release: pick a model from the tiles (the first by default); it is drawn at every batch,
    // one bar per release (oldest first, the newest in the brand red), the newest over the one before above each batch.
    const sorted = rows;  // best total first, like the table
    const relsOf = (r) => vers.filter((v) => runsOf[r.key][v]), sub = (r) => 'RNGD ×' + r.n + ' · ' + relsOf(r).map(short).join(', ');
    const items = sorted.map((r) => ({ key: r.key, name: r.model, sub: sub(r) })), pick = sorted.length ? pickedKey(st, 'pick', items, items[0].key) : null;
    const r0 = sorted.filter((r) => r.key === pick)[0], rv = r0 ? relsOf(r0) : [];
    const series = rv.map((v, j) => ({ name: short(v), color: 'var(--c-red-' + Math.min(3, rv.length - 1 - j) + ')', run: runsOf[r0.key][v] }));
    const list = !r0 ? '' : pickTiles('pick', items, pick) + '<div class="picked"><h3>' + esc(r0.model) + ' <span class="faint mono">' + esc(sub(r0)) + '</span></h3>'
      + batchChart(M, series, [series.length - 1, series.length - 2], isl, osl) + '</div>';
    return page('<section class="panel">' + head()
      + (list || '<div class="empty">No RNGD model was measured on two or more releases for this selection.</div>') + more + '</section>');
  }
  return page('<section class="panel">' + head('<div class="heat-scale"><span class="mono">0.80×</span><span class="ramp"></span><span class="mono">1.20×</span><span>vs previous release</span></div>')
    + (rows.length ? '<div class="scroll"><div class="heat" role="table" style="grid-template-columns:minmax(220px,1.6fr) 56px repeat(' + vers.length + ',minmax(110px,1fr)) 120px">' + cells + '</div></div>'
      : '<div class="empty">No RNGD model was measured on two or more releases for this selection.</div>')
    + more + '</section>');
};
