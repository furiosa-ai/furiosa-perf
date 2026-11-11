// RNGD vs GPU (#summary): RNGD against one GPU platform, one metric per model at one ISL/OSL pair.
'use strict';

PAGES.summary = async function (st) {
  const all = await generationRuns();
  const tasks = Object.keys(GEN_TASKS).filter((t) => all.some((r) => r.task === t));
  const task = tasks.indexOf(st.task) >= 0 ? st.task : tasks[0];
  const tRuns = all.filter((r) => r.task === task);
  const scens = shapeOptions(tRuns);
  const scen = pickShape(st.scen, scens);
  const slo = st.slo > 0 ? st.slo : 20, rackKw = +st.rack > 0 ? +st.rack : RACK_KW;

  // Platforms that measured this ISL/OSL pair.
  const plats = {};
  tRuns.forEach((r) => { if ((r.shapes || []).indexOf(scen) >= 0) plats[platformKey(r)] = r; });
  const keys = Object.keys(plats);
  const rngdKeys = keys.filter((k) => family(plats[k].hw) === 'rngd').sort((a, b) => cmpVer(plats[b].ver, plats[a].ver));
  const gpuKeys = keys.filter((k) => family(plats[k].hw) !== 'rngd')
    .sort((a, b) => famOrder(plats[a].hw) - famOrder(plats[b].hw) || cmpVer(plats[b].ver, plats[a].ver));
  const rngd = rngdKeys.indexOf(st.rngd) >= 0 ? st.rngd : rngdKeys[0];
  // GPU side: a device ('hw:<hw>', each model at the newest version that ran it - a model the newest
  // vLLM cannot serve, like a vendor fork, still gets compared) or one exact platform (one version for all).
  const gpuHws = [...new Set(gpuKeys.map((k) => plats[k].hw))];
  const gpuOpts = gpuHws.map((h) => 'hw:' + h).concat(gpuKeys);
  // RTX PRO 6000 by default, the GPU the published efficiency charts compare against.
  const gpu = gpuOpts.indexOf(st.gpu) >= 0 ? st.gpu : 'hw:' + (gpuHws.filter((h) => family(h) === 'rtx')[0] || gpuHws[0]);
  const gpuHw = gpu && gpu.indexOf('hw:') === 0 ? gpu.slice(3) : null;
  const gpuLabel = !gpu ? '—' : gpuHw ? gpuHw + ' · newest version per model' : platformLabel(gpu);
  // Devices, one count per model and side whatever the metric: RNGD's count on both sides (default), each
  // side's best by TPS per kW at the SLO, or one count on both sides.
  const counts = [...new Set(tRuns.map((r) => r.n))].sort((a, b) => a - b);
  const dev = counts.indexOf(+st.dev) >= 0 ? +st.dev : st.dev === 'best' ? 'best' : 'match';
  // How many models to show (best ratio first), so a meeting can stop at the strong ones.
  const featured = featuredFor((plats[rngd] || {}).ver);
  const show = showOf(st, featured);
  const metric = METRICS[st.metric] ? st.metric : 'tpkw';
  const M = METRICS[metric];

  const opt = (k) => ({ v: k, label: platformLabel(k) });
  // Grouped controls: a sticky side panel on wide screens, a block above the page on narrow ones (style.css).
  const controls = ''
    + sideGroup('Compare', select('RNGD', 'rngd', rngd, rngdKeys.map(opt))
    + select('Compare with', 'gpu', gpu, gpuOpts.map((k) => (k.indexOf('hw:') === 0 ? { v: k, label: k.slice(3) + ' · newest per model' } : opt(k))))
    + select('Devices', 'dev', String(dev), [{ v: 'match', label: 'Same as RNGD' }, { v: 'best', label: 'Best per side · TPS/kW' }].concat(counts.map((n) => ({ v: String(n), label: '×' + n + ' both sides' })))))
    + sideGroup('Scenario', select('Task', 'task', task, tasks.map((t) => ({ v: t, label: GEN_TASKS[t] })))
      + select('ISL / OSL', 'scen', scen, scens) + (M.slo ? sloInput(slo) : '') + (M.rack ? rackInput(rackKw) : '')
      + (M.kw ? serverKwInputs(['rngd', family(gpuHw || (plats[gpu] || {}).hw || '')]) : ''))
    + sideGroup('View', select('Metric', 'metric', metric, Object.keys(METRICS).map((k) => ({ v: k, label: METRICS[k].label })))
      + showSelect(show, featured));
  const page = (body) => sidePage('RNGD vs GPU', controls, body);
  if (!rngd || !scen) return page('<div class="status">No RNGD results for this task.</div>');

  const onGpu = (r) => (gpuHw ? r.hw === gpuHw && family(r.hw) !== 'rngd' && (r.shapes || []).indexOf(scen) >= 0 : platformKey(r) === gpu);
  const newestVer = {};  // per model, the newest version measured on the compared GPU
  tRuns.filter(onGpu).forEach((r) => { if (!newestVer[r.model] || cmpVer(r.ver, newestVer[r.model]) > 0) newestVer[r.model] = r.ver; });
  const hwNewest = Object.values(newestVer).reduce((a, v) => (!a || cmpVer(v, a) > 0 ? v : a), null);
  const sel = tRuns.filter((r) => (platformKey(r) === rngd || (onGpu(r) && r.ver === newestVer[r.model])) && (typeof dev !== 'number' || r.n === dev));
  await loadPoints(sel);
  const [isl, osl] = scen.split('/').map(Number);
  const sideOf = (r) => (platformKey(r) === rngd ? 'r' : 'g');
  // The device count of each model and side, chosen once so that every metric compares the same runs.
  const ns = {};
  sel.forEach((r) => { const s = (ns[r.model] = ns[r.model] || { r: [], g: [] }); if (s[sideOf(r)].indexOf(r.n) < 0) s[sideOf(r)].push(r.n); });
  const pickN = {};
  Object.keys(ns).forEach((m) => {
    const rs = ns[m].r.slice().sort((a, b) => b - a), gs = ns[m].g;
    if (dev === 'match') {
      // RNGD: its largest count the GPU also measured; GPU: that count, else the nearest one measured (marked).
      const rn = rs.filter((n) => gs.indexOf(n) >= 0)[0] || rs[0];
      const gn = gs.slice().sort((a, b) => Math.abs(Math.log(a / rn)) - Math.abs(Math.log(b / rn)) || b - a)[0];
      pickN[m] = { r: rn, g: gn };
    } else if (dev === 'best') {
      const by = (side) => sel.filter((r) => r.model === m && sideOf(r) === side)
        .map((r) => ({ n: r.n, v: (METRICS.tpkw.f(pointsAt(r, isl, osl), r, slo) || { v: -1 }).v }))
        .sort((a, b) => b.v - a.v || a.n - b.n)[0];
      pickN[m] = { r: (by('r') || {}).n, g: (by('g') || {}).n };
    } else pickN[m] = { r: dev, g: dev };
  });
  const best = {};
  sel.forEach((r) => {
    const side = sideOf(r);
    if (!pickN[r.model] || r.n !== pickN[r.model][side]) return;
    const m = M.f(pointsAt(r, isl, osl), r, slo, rackKw);
    if (m) (best[r.model] = best[r.model] || {})[side] = { v: m.v, n: r.n, ge: !!m.ge, run: r };
  });
  // ratio > 1 always means RNGD is better, whichever direction the metric improves in.
  // Only models both sides measured: a model without the compared GPU has nothing to compare.
  const list = Object.keys(best).filter((m) => best[m].r && best[m].g).map((m) => {
    const r = best[m].r, g = best[m].g;
    // Per-batch metrics: the batches both sides measured, averaged (batchRatio); the others: their one value.
    const perBatch = M.point ? batchRatio(M, isl, osl, g.run, r.run) : null;
    const ratio = perBatch !== null ? perBatch : !g || !(g.v > 0) || !(r.v > 0) ? null : M.higher ? r.v / g.v : g.v / r.v;
    return { model: m, r: r, g: g, ratio: ratio };
  });
  list.sort((a, b) => (b.ratio === null ? -1 : b.ratio) - (a.ratio === null ? -1 : a.ratio) || a.model.localeCompare(b.model));
  const total = list.length;
  applyShow(list, show, featured);
  const more = moreButton(list.length, total, show);
  // Bar scale: the GPU bar has one fixed length on every row and the RNGD bar is that length × the ratio,
  // so a row reads as RNGD's multiple of the GPU (the best ratio shown reaches 85% of the track).
  const maxRatio = Math.max(1, ...list.map((x) => x.ratio || 0));
  const gpuLen = Math.min(60, Math.max(25, 85 / maxRatio));

  const label = (x) => (x ? (x.ge ? '≥' : '') + fmt(x.v) + '  ×' + x.n : '—');
  // No one value (M.point): pick a model from the tiles (the first by default); it is drawn at every batch.
  const sub = (x) => 'RNGD ×' + x.r.n + ' · ' + x.g.run.hw + ' ×' + x.g.n + (dev === 'match' && x.g.n !== x.r.n ? ' (×' + x.r.n + ' not measured)' : '')
    + (gpuHw ? ' · ' + backendLabel(x.g.run.backend) + ' ' + x.g.run.ver : '');
  const items = list.map((x) => ({ key: x.model, name: x.model, sub: sub(x) + (x.ratio === null ? '' : ' · ' + multText(x.ratio, M.higher)) })), pick = M.point && list.length ? pickedKey(st, 'pick', items, items[0].key) : null;
  const x0 = list.filter((x) => x.model === pick)[0];
  const byBatch = !x0 ? '' : pickTiles('pick', items, pick) + '<div class="picked"><h3>' + esc(x0.model) + ' <span class="faint mono">' + esc(sub(x0)) + '</span></h3>'
    + batchChart(M, [{ name: 'RNGD ×' + x0.r.n, color: 'var(--c-red-0)', run: x0.r.run }, { name: x0.g.run.hw + ' ×' + x0.g.n + ' · ' + x0.g.run.ver, color: 'var(--g-a)', run: x0.g.run }], [0, 1], isl, osl) + '</div>';
  const rows = M.point ? '' : list.map((x) => {
    // GPU bar fixed, RNGD bar = GPU × ratio (capped at 88% so the multiplier still fits; values are on the right).
    const gp = x.ratio === null ? 0 : gpuLen, rp = x.ratio === null ? 0 : Math.max(0.2, Math.min(88, gpuLen * x.ratio));
    let ratio = '';
    if (x.ratio !== null) {
      // A sweep that hit its top concurrency only bounds the value, so the ratio is a bound too.
      const bound = x.g.ge && !x.r.ge ? '≤' : x.r.ge && !x.g.ge ? '≥' : '';
      const top = rp <= gp ? 6 : 30;  // on the shorter side's row, just past the longer bar
      ratio = '<div class="ratio" style="top:' + top + 'px;left:calc(' + Math.max(rp, gp) + '% + 8px);color:' + (x.ratio >= 1 ? 'var(--accent)' : 'var(--title)') + '">' + bound + multText(x.ratio, M.higher) + '</div>';
    }
    const nearest = x.g.n !== x.r.n && dev === 'match' ? ' <span class="faint mono" title="the GPU was not measured at RNGD\'s device count">· GPU ×' + x.r.n + ' not measured</span>' : '';
    const older = gpuHw && x.g.run.ver !== hwNewest ? ' <span class="faint mono" title="not measured on the newest version">· ' + esc(backendLabel(x.g.run.backend) + ' ' + x.g.run.ver) + '</span>' : '';
    return '<div class="bar-row"><div class="name">' + esc(x.model) + older + nearest + '</div><div class="bar-track">'
      + '<div class="bar r" style="width:' + rp + '%"></div><div class="bar g" style="width:' + gp + '%"></div>' + ratio
      + '</div><div class="bar-vals"><span>' + label(x.r) + '</span><span class="muted">' + label(x.g) + '</span></div></div>';
  }).join('');

  // Panel heading: what is measured in plain words, then one line of how (conditions, power basis, devices).
  const gHw = gpuHw || (plats[gpu] || {}).hw || '';
  const servers = M.kw ? ' 8-card server power: RNGD ' + serverKw('RNGD') + ' kW' + (SERVER_KW[family(gHw)] ? ', ' + FAMILY_NAME[family(gHw)] + ' ' + serverKw(gHw) + ' kW' : '') + '.' : '';
  const devices = dev === 'match' ? ' Both sides at RNGD\'s device count (the GPU at its nearest measured count where it lacks it, marked).'
    : dev === 'best' ? ' Each side at its best device count by TPS per kW at the SLO, the same for every metric.' : ' ×' + dev + ' on both sides.';
  const bound = M.slo ? ' ≥: still above ' + slo + ' tok/s per user at the largest batch measured.' : '';
  const HEAD = metricText(metric, M, slo, rackKw);
  if (M.point) HEAD[1] += ' Ranked at ' + pointRead(metric) + '; pick a model to see every batch.';
  const headSub = GEN_TASKS[task] + ', ' + scen + ' ISL/OSL.' + servers + devices + (M.slo ? ' ≥ = lower bound.' : '');
  // Fixed title: what is compared lives in the controls, and in the panel heading and legend below.
  return page(''
    + '<section class="panel"><div class="panel-head"><div><h2>' + esc(HEAD[0]) + '</h2><div class="panel-sub">' + note('summary', headSub, HEAD[1] + bound) + '</div></div><div class="legend">'
    + '<span><span class="swatch" style="background:var(--c-red-0)"></span>' + esc(platformLabel(rngd)) + '</span>'
    + '<span><span class="swatch" style="background:var(--g-a)"></span>' + esc(gpuLabel) + '</span></div></div>'
    + ((M.point ? byBatch : rows) || '<div class="empty">No RNGD points for this task and ISL/OSL.</div>') + more
    + '</section>');
};

