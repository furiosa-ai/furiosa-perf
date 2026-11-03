// Variants: a measurement run again with server options changed (--server-arg), next to the run without them,
// e.g. a chat scenario with prefix caching off against the preset's prefix caching on.
'use strict';

// The load a scenario steps through: summary column and the run's capacity field.
const LOADS = [
  { col: 'Users', cap: 'max_users', label: 'users' },
  { col: 'Trace Speed(%)', cap: 'max_speed_pct', label: '% trace speed' },
];
// Point values compared at one load: column, label, higher is better.
const AT_LOAD = [
  ['Output Throughput(tok/s)', 'Output TPS', true],
  ['P95 TTFT(ms)', 'TTFT p95 (ms)', false],
  ['P95 TPOT(ms)', 'TPOT p95 (ms)', false],
  ['prefix_hit_rate', 'Prefix hit', true],
];

function delta(v, base, higher) {
  if (!(v > 0) || !(base > 0) || v === base) return '';
  const pct = (v / base - 1) * 100, better = higher ? pct > 0 : pct < 0;
  return ' <span style="color:' + (better ? 'var(--accent)' : 'var(--muted)') + '">' + (pct > 0 ? '+' : '') + pct.toFixed(0) + '%</span>';
}

PAGES.variants = async function () {
  const groups = {};
  latestRuns((await loadIndex()).filter((r) => r.kind === 'scenario')).forEach((r) => {
    const k = [r.hw, r.n, r.layout, r.backend, r.ver, r.task, r.model].join('|');
    (groups[k] = groups[k] || []).push(r);
  });
  const sets = Object.values(groups).filter((g) => g.length > 1);
  const head = '<div class="head"><div><h1>Variants</h1><div class="sub">The same scenario with server options changed, against the run without them</div></div></div>';
  if (!sets.length) {
    return head + '<div class="status">No variants yet. Run a scenario again with a server option changed, e.g.<br><br>'
      + '<code class="mono">furiosa-perf run scenario chat-synthetic --server-config … --server-arg no-enable-prefix-caching</code></div>';
  }
  await loadPoints([].concat(...sets));

  const panels = sets.map((g) => {
    g.sort((a, b) => (a.variant ? 1 : 0) - (b.variant ? 1 : 0) || a.variant.localeCompare(b.variant));
    const base = g[0].variant ? null : g[0];
    const pts = (r) => (DATA.points[r.id] || []).filter((p) => p.ok);
    const load = LOADS.filter((l) => g.every((r) => pts(r).some((p) => p.raw[l.col] !== undefined)))[0] || LOADS[0];
    // The highest load every run measured validly: the values there are compared like for like.
    const common = pts(g[0]).map((p) => p.raw[load.col]).filter((v) => g.every((r) => pts(r).some((p) => p.raw[load.col] === v)));
    const at = common.length ? Math.max.apply(null, common) : null;
    const pointAt = (r) => pts(r).filter((p) => p.raw[load.col] === at)[0];
    const capOf = (r) => (r.capacity || {})[load.cap];

    const rows = g.map((r) => {
      const p = pointAt(r), bp = base && pointAt(base);
      return '<tr><td>' + (r.variant ? esc(r.variant) : '<span class="muted">as configured</span>') + '</td>'
        + '<td class="num">' + fmt(capOf(r)) + (base && r !== base ? delta(capOf(r), capOf(base), true) : '') + '</td>'
        + AT_LOAD.map((m) => '<td class="num">' + (p ? fmt(p.raw[m[0]]) + (bp && r !== base ? delta(p.raw[m[0]], bp.raw[m[0]], m[2]) : '') : '—') + '</td>').join('')
        + '<td class="date">' + day(r.date) + '</td></tr>';
    }).join('');
    return '<section class="panel"><div class="panel-head"><h2>' + esc(g[0].model + ' · ' + g[0].task) + '</h2><span class="muted">' + esc(seriesLabel(g[0])) + '</span></div>'
      + '<div class="scroll"><table class="list"><thead><tr><th>Server options</th><th class="num">Max ' + esc(load.label) + ' within SLO</th>'
      + AT_LOAD.map((m) => '<th class="num">' + esc(m[1]) + (at === null ? '' : ' @ ' + fmt(at)) + '</th>').join('')
      + '<th>Measured</th></tr></thead><tbody>' + rows + '</tbody></table></div></section>';
  }).join('');
  return head + panels + '<p class="caption">Percentages are against the run without server options changed (as configured: server config plus preset). '
    + 'Point values are taken at the highest load every run of the group measured validly (@), so they compare like for like. '
    + 'Variants are made with <code class="mono">--server-arg</code>; only scenario runs are shown here, every run is on the Data page.</p>';
};
