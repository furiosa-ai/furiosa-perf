// Summary: RNGD against one GPU platform, one metric per model at one ISL/OSL pair.
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
  // Devices: each side's best device count per model, or the same count on both sides.
  const counts = [...new Set(tRuns.map((r) => r.n))].sort((a, b) => a - b);
  const dev = counts.indexOf(+st.dev) >= 0 ? +st.dev : 'best';
  const metric = METRICS[st.metric] ? st.metric : 'tpkw';
  const M = METRICS[metric];

  const opt = (k) => ({ v: k, label: platformLabel(k) });
  const controls = '<div class="controls">'
    + select('Task', 'task', task, tasks.map((t) => ({ v: t, label: GEN_TASKS[t] })))
    + select('ISL / OSL', 'scen', scen, scens)
    + select('RNGD', 'rngd', rngd, rngdKeys.map(opt))
    + select('Compare with', 'gpu', gpu, gpuOpts.map((k) => (k.indexOf('hw:') === 0 ? { v: k, label: k.slice(3) + ' · newest per model' } : opt(k))))
    + select('Devices', 'dev', String(dev), [{ v: 'best', label: 'Best per side' }].concat(counts.map((n) => ({ v: String(n), label: '×' + n + ' both sides' }))))
    + select('Metric', 'metric', metric, Object.keys(METRICS).map((k) => ({ v: k, label: METRICS[k].label })))
    + (M.slo ? sloInput(slo) : '') + (M.rack ? rackInput(rackKw) : '') + '</div>';
  if (!rngd || !scen) return '<div class="head"><h1>Summary</h1>' + controls + '</div><div class="status">No RNGD results for this task.</div>';

  const onGpu = (r) => (gpuHw ? r.hw === gpuHw && family(r.hw) !== 'rngd' && (r.shapes || []).indexOf(scen) >= 0 : platformKey(r) === gpu);
  const newestVer = {};  // per model, the newest version measured on the compared GPU
  tRuns.filter(onGpu).forEach((r) => { if (!newestVer[r.model] || cmpVer(r.ver, newestVer[r.model]) > 0) newestVer[r.model] = r.ver; });
  const hwNewest = Object.values(newestVer).reduce((a, v) => (!a || cmpVer(v, a) > 0 ? v : a), null);
  const sel = tRuns.filter((r) => (platformKey(r) === rngd || (onGpu(r) && r.ver === newestVer[r.model])) && (dev === 'best' || r.n === dev));
  await loadPoints(sel);
  const [isl, osl] = scen.split('/').map(Number);
  // Best device count per model and side.
  const best = {};
  sel.forEach((r) => {
    const m = M.f(pointsAt(r, isl, osl), r, slo, rackKw);
    if (!m) return;
    const slot = (best[r.model] = best[r.model] || {}), side = platformKey(r) === rngd ? 'r' : 'g', cur = slot[side];
    if (!cur || (M.higher ? m.v > cur.v : m.v < cur.v)) slot[side] = { v: m.v, n: r.n, ge: !!m.ge, run: r };
  });
  // ratio > 1 always means RNGD is better, whichever direction the metric improves in.
  // Only models both sides measured: a model without the compared GPU has nothing to compare.
  const list = Object.keys(best).filter((m) => best[m].r && best[m].g).map((m) => {
    const r = best[m].r, g = best[m].g;
    const ratio = !g || !(g.v > 0) || !(r.v > 0) ? null : M.higher ? r.v / g.v : g.v / r.v;
    return { model: m, r: r, g: g, ratio: ratio };
  });
  list.sort((a, b) => (M.point ? 0 : (b.ratio === null ? -1 : b.ratio) - (a.ratio === null ? -1 : a.ratio)) || a.model.localeCompare(b.model));

  const label = (x) => (x ? (x.ge ? '≥' : '') + fmt(x.v) + '  ×' + x.n : '—');
  // No one value (M.point): pick a model from the tiles (the first by default); it is drawn at every batch.
  const sub = (x) => 'RNGD ×' + x.r.n + ' · ' + x.g.run.hw + ' ×' + x.g.n + (gpuHw ? ' · ' + backendLabel(x.g.run.backend) + ' ' + x.g.run.ver : '');
  const items = list.map((x) => ({ key: x.model, name: x.model, sub: sub(x) })), pick = M.point && list.length ? pickedKey(st, 'pick', items, items[0].key) : null;
  const x0 = list.filter((x) => x.model === pick)[0];
  const byBatch = !x0 ? '' : pickTiles('pick', items, pick) + '<div class="picked"><h3>' + esc(x0.model) + ' <span class="faint mono">' + esc(sub(x0)) + '</span></h3>'
    + batchChart(M, [{ name: 'RNGD ×' + x0.r.n, color: 'var(--c-red-0)', run: x0.r.run }, { name: x0.g.run.hw + ' ×' + x0.g.n + ' · ' + x0.g.run.ver, color: 'var(--g-a)', run: x0.g.run }], [0, 1], isl, osl) + '</div>';
  const rows = M.point ? '' : list.map((x) => {
    // Each row is scaled to its own longer bar (models differ in size by 10× or more; the values are on the right).
    // Bars use up to 85% of the track so the multiplier written past the longer bar still fits.
    const rowMax = Math.max(x.r.v, x.g ? x.g.v : 0);
    const rp = rowMax > 0 ? Math.max(0.2, x.r.v / rowMax * 85) : 0, gp = x.g && rowMax > 0 ? Math.max(0.2, x.g.v / rowMax * 85) : 0;
    let ratio = '';
    if (x.ratio !== null) {
      // A sweep that hit its top concurrency only bounds the value, so the ratio is a bound too.
      const bound = x.g.ge && !x.r.ge ? '≤' : x.r.ge && !x.g.ge ? '≥' : '';
      const top = rp <= gp ? 6 : 30;  // on the shorter side's row, just past the longer bar
      ratio = '<div class="ratio" style="top:' + top + 'px;left:calc(' + Math.max(rp, gp) + '% + 8px);color:' + (x.ratio >= 1 ? 'var(--accent)' : 'var(--title)') + '">' + bound + x.ratio.toFixed(2) + '×</div>';
    }
    const older = gpuHw && x.g.run.ver !== hwNewest ? ' <span class="faint mono" title="not measured on the newest version">· ' + esc(backendLabel(x.g.run.backend) + ' ' + x.g.run.ver) + '</span>' : '';
    return '<div class="bar-row"><div class="name">' + esc(x.model) + older + '</div><div class="bar-track">'
      + '<div class="bar r" style="width:' + rp + '%"></div><div class="bar g" style="width:' + gp + '%"></div>' + ratio
      + '</div><div class="bar-vals"><span>' + label(x.r) + '</span><span class="muted">' + label(x.g) + '</span></div></div>';
  }).join('');

  const compared = list.filter((x) => x.ratio !== null);
  const med = median(compared.map((x) => x.ratio));
  const title = (M.title || M.label) + metricBasis(M, slo, rackKw) + (M.bestN ? (dev === 'best' ? ' · best device count' : ' · ×' + dev + ' both sides') : '') + (M.higher ? '' : ' · lower is better') + (M.point ? ' · pick a model for every batch' : '');
  const servers = '8-card server power (RNGD 3.0 kW, RTX PRO 6000 6.5 kW, H100/H200 10.2 kW assumed)';
  const bound = ' ≥ marks a system still faster than ' + slo + ' TPS per user at its largest measured batch.';
  const caption = metric === 'tpkw' ? 'Tokens per second per kW at ' + slo + ' TPS per user, ' + scen + ' ISL/OSL. Per-device throughput scaled to ' + servers + '.' + bound
    : metric === 'rack' ? 'Users at ≥ ' + slo + ' TPS per user that a ' + rackKw + ' kW rack serves, ' + scen + ' ISL/OSL: whole servers per rack from ' + servers + ', × users per server.' + bound
    : M.slo ? 'Users at ≥ ' + slo + ' TPS per user, scaled to an 8-card server (× 8 / devices measured).' + bound : '';
  return '<div class="head"><div><h1>' + esc(platformLabel(rngd)) + '</h1><div class="sub">vs ' + esc(gpuLabel) + ' · '
    + esc(GEN_TASKS[task].toLowerCase() + ' ' + scen + metricBasis(M, slo, rackKw)) + '</div></div>' + controls + '</div>'
    + '<section class="kpis">' + kpi('Models compared', M.point ? list.length : compared.length)
    + (M.point ? '' : kpi('RNGD ahead', compared.filter((x) => x.ratio >= 1).length + ' / ' + compared.length, 'var(--accent)') + kpi('Median RNGD advantage', med === null ? '—' : med.toFixed(2) + '×')) + '</section>'
    + '<section class="panel"><div class="panel-head"><h2>' + esc(title) + '</h2><div class="legend">'
    + '<span><span class="swatch" style="background:var(--c-red-0)"></span>' + esc(platformLabel(rngd)) + '</span>'
    + '<span><span class="swatch" style="background:var(--g-a)"></span>' + esc(gpuLabel) + '</span></div></div>'
    + ((M.point ? byBatch : rows) || '<div class="empty">No RNGD points for this task and ISL/OSL.</div>')
    + (caption ? '<p class="caption">' + esc(caption) + '</p>' : '') + '</section>';
};
