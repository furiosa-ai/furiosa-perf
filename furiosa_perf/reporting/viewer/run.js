// Run details: how one run was measured. The server command, the client command of every batch (the selected
// batch's full command, then what each batch changed), what the client recorded, and the environment.
// Opened from the Explorer's legend and the Data page (#run?id=…&scen=…).
'use strict';

// A command as one would type it anywhere: the tool by name rather than the measuring host's path
// ('…/python -m vllm.entrypoints.cli.main' is 'vllm', '/root/venv/bin/vllm' is 'vllm'), and without the
// host's result directory. The recorded command is kept as is in the data.
function portable(command) {
  return command.replace(/^\S*python[\d.]*\s+-m\s+vllm\.entrypoints\.cli\.main\b/, 'vllm')
    .replace(/^\S*\/(?=\S+(\s|$))/, '')
    .replace(/\s--result-dir(=|\s+)\S+/, '');
}

// "--flag value" pairs kept together, so a batch's changed options read as options.
function argsOf(command) {
  const t = command.split(/\s+/).filter(Boolean), out = [];
  for (let i = 0; i < t.length; i++) out.push(t[i].startsWith('--') && t[i + 1] && !t[i + 1].startsWith('-') && t[i].indexOf('=') < 0 ? t[i] + ' ' + t[++i] : t[i]);
  return out;
}
// One option per line, for reading.
function shellLines(args) { return args.map((a, i) => (i && a.startsWith('-') ? ' \\\n    ' : i ? ' ' : '') + a).join(''); }

// Clicking a batch shows its full command in the box above the table (#run?…&b=<batch>).
document.addEventListener('click', (e) => {
  const tr = e.target.closest && e.target.closest('tr.cmd-pick');
  if (tr) setState({ b: tr.dataset.b });
});

function copyButton(text, label) { return '<button class="btn" data-copy="' + esc(text) + '">' + esc(label || 'Copy') + '</button>'; }

function envRows(env) {
  return '<dl class="kv">' + Object.keys(env).map((k) => {
    const v = env[k];
    return '<dt>' + esc(k) + '</dt><dd>' + (v !== null && typeof v === 'object' ? '<pre class="json">' + esc(JSON.stringify(v, null, 2)) + '</pre>' : esc(String(v))) + '</dd>';
  }).join('') + '</dl>';
}

PAGES.run = async function (st) {
  const run = (await loadIndex()).filter((r) => r.id === st.id)[0];
  if (!run) return '<div class="status">No run ' + esc(st.id || '') + ' in this site.</div>';
  await loadPoints([run]);
  const d = DATA.details[run.id] || {}, commands = d.commands || {}, params = d.params || {};
  const shapes = run.shapes || [], scen = shapes.indexOf(st.scen) >= 0 ? st.scen : shapes.indexOf('1024/1024') >= 0 ? '1024/1024' : shapes[0];
  const [isl, osl] = (scen || '0/0').split('/').map(Number);
  const pts = pointsAt(run, isl, osl), key = (p) => isl + '/' + osl + '/' + p.conc;

  // Server
  const server = portable(d.server_command || '');
  const flags = server.split(/\s+/).filter((a) => /^--(enable|disable|experimental|no-enable)/.test(a));
  const serverPanel = '<section class="panel"><div class="panel-head"><h2>Server</h2>' + (server ? copyButton(server, 'Copy command') : '') + '</div>'
    + (server ? '<pre class="cmd">' + esc(shellLines(argsOf(server))) + '</pre>' : '<div class="empty">The server command was not recorded.</div>')
    + (flags.length ? '<div class="chips"><span class="muted">Feature flags</span>' + flags.map((f) => '<code>' + esc(f) + '</code>').join('') + '</div>' : '') + '</section>';

  // Client: the options every batch of this ISL/OSL shares, then each batch's own.
  const cmdOf = (p) => (commands[key(p)] ? portable(commands[key(p)]) : '');
  const cmds = pts.map(cmdOf).filter(Boolean), lists = cmds.map(argsOf);
  const shared = lists.length ? lists[0].filter((a) => lists.every((l) => l.indexOf(a) >= 0)) : [];
  const pcols = [...new Set(pts.flatMap((p) => Object.keys(params[key(p)] || {})))];
  // The selected batch (the first with a command by default) has its full command shown above the table.
  const withCmd = pts.filter(cmdOf), picked = withCmd.filter((p) => String(p.conc) === st.b)[0] || withCmd[0];
  const rows = pts.map((p) => {
    const c = cmdOf(p), own = c ? argsOf(c).filter((a) => shared.indexOf(a) < 0) : [], pr = params[key(p)] || {};
    return '<tr class="' + (p.ok ? '' : 'faint') + (c ? ' cmd-pick' + (p === picked ? ' sel' : '') : '') + '"' + (c ? ' data-b="' + p.conc + '" title="Show the full command"' : '') + '><td class="num">b' + p.conc + '</td>'
      + (cmds.length ? '<td class="mono own">' + (c ? esc(own.join('  ')) : '<span class="faint">not recorded</span>') + '</td>' : '')
      + pcols.map((k) => '<td class="num">' + esc(pr[k] === undefined ? '' : typeof pr[k] === 'number' && !Number.isInteger(pr[k]) ? fmt(pr[k]) : String(pr[k])) + '</td>').join('') + '</tr>';
  }).join('');
  const missing = pts.length - cmds.length;
  const clientPanel = '<section class="panel"><div class="panel-head"><h2>Benchmark</h2><span class="muted">' + esc(run.client || (cmds.length && /bench serve/.test(cmds[0]) ? 'vllm bench serve' : '')) + '</span></div>'
    + pickTiles('scen', shapes.map((s) => ({ key: s, name: s, sub: 'ISL / OSL' })), scen)
    + (picked ? '<div class="cmd-head"><span class="muted">Full command of b' + picked.conc + ' · ' + esc(scen) + ' — click a batch below to switch</span>' + copyButton(cmdOf(picked)) + '</div><pre class="cmd">' + esc(shellLines(argsOf(cmdOf(picked)))) + '</pre>'
      : '<div class="note">This run did not record its client command; the table shows what the client recorded about each batch instead.</div>')
    + (missing > 0 && cmds.length ? '<div class="note">' + missing + ' of ' + pts.length + ' batches have no recorded command.</div>' : '')
    + '<div class="scroll"><table class="list"><thead><tr><th class="num">Batch</th>' + (cmds.length ? '<th>Options of this batch</th>' : '')
    + pcols.map((k) => '<th class="num">' + esc(k) + '</th>').join('') + '</tr></thead><tbody>' + rows + '</tbody></table></div></section>';

  const env = d.environment || {};
  const envPanel = '<section class="panel"><div class="panel-head"><h2>Environment</h2></div>'
    + (Object.keys(env).length ? envRows(env) : '<div class="empty">No environment was recorded.</div>') + '</section>';

  const back = '#explorer?model=' + encodeURIComponent(run.model) + '&task=' + encodeURIComponent(run.task) + (scen ? '&scen=' + encodeURIComponent(scen) : '');
  return '<div class="run-page"><div class="muted" style="font-size:13px"><a href="' + back + '">← Explorer · ' + esc(run.model) + '</a></div>'
    + '<div><h1>' + esc(run.model_id) + '</h1><div class="sub"><span style="color:var(--c-' + hueOf(run.hw) + '-0)">●</span> '
    + esc(hwName(run.hw) + ' ×' + run.n + (run.layout ? ' ' + run.layout : '') + ' · ' + backendLabel(run.backend) + ' ' + run.ver + ' · ' + run.kind + ' · ' + run.task
      + (run.variant ? ' · ' + run.variant : '') + ' · measured ' + (run.date || '').replace('T', ' ').slice(0, 16) + ' UTC') + '</div></div>'
    + serverPanel + clientPanel + envPanel + '</div>';
};

// Copy buttons: the text in data-copy.
document.addEventListener('click', (e) => {
  const b = e.target.closest && e.target.closest('[data-copy]');
  if (!b) return;
  navigator.clipboard.writeText(b.dataset.copy).then(() => { const t = b.textContent; b.textContent = 'Copied'; setTimeout(() => { b.textContent = t; }, 1200); });
});
