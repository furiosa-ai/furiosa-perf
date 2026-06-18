(() => {
  // ============================================================
  // Shared helpers
  // ============================================================

  // Max concurrency whose per-user throughput (TPS/User) still meets `slo`: the right
  // edge of the feasible region {Concurrent : TPS/User >= slo}, by piecewise-linear
  // interpolation over the *measured* points. Robust to non-monotonic curves where
  // TPS/User rises before it falls -> if no sample reaches `slo`, nothing is feasible
  // (returns 0), and we take the *largest* crossing so a rising left segment can't
  // produce a spurious solution. Mirrors `interp_conc_from_tps_user` in schemas.py.
  function interpConcFromTpsUser(conc, tps, slo) {
    if (!Array.isArray(conc) || !Array.isArray(tps) || conc.length === 0 || !(slo > 0)) return 0;

    const idx = conc.map((_, i) => i).filter(i => Number.isFinite(conc[i]) && Number.isFinite(tps[i]));
    idx.sort((a, b) => conc[a] - conc[b]);
    const c = idx.map(i => conc[i]);
    const t = idx.map(i => tps[i]);

    if (c.length === 0) return 0;
    if (!t.some(v => v >= slo)) return 0;            // SLO above the achievable peak
    if (c.length === 1) return Math.max(0, c[0]);

    let best = null;
    for (let i = 0; i < c.length - 1; i++) {
      const y0 = t[i], y1 = t[i + 1];
      if (y0 === y1) continue;
      if ((y0 - slo) * (y1 - slo) <= 0) {            // slo lies within this segment
        const f = (slo - y0) / (y1 - y0);
        const xc = c[i] + f * (c[i + 1] - c[i]);
        best = best === null ? xc : Math.max(best, xc);
      }
    }

    // Feasible to the highest measured concurrency: extrapolate along a decreasing tail.
    if (t[t.length - 1] >= slo) {
      const y0 = t[t.length - 2], y1 = t[t.length - 1];
      const x0 = c[c.length - 2], x1 = c[c.length - 1];
      if (y1 < y0) {
        const f = (slo - y0) / (y1 - y0);
        const xc = x0 + f * (x1 - x0);
        best = best === null ? xc : Math.max(best, xc);
      } else {
        best = best === null ? c[c.length - 1] : Math.max(best, c[c.length - 1]);
      }
    }

    return best === null ? 0 : Math.max(0, best);
  }

  // Per-device max concurrency for a given SLO, read straight from the interactive
  // plot's embedded measured curves (meta.curveConc / meta.curveTpsU).
  function usersAtSlo(gd, slo) {
    const meta = (gd && gd._fullLayout && gd._fullLayout.meta) || {};
    const curveConc = meta.curveConc || [];
    const curveTpsU = meta.curveTpsU || [];
    return curveConc.map((c, i) => interpConcFromTpsUser(c, curveTpsU[i], slo));
  }

  function findParentGroup(el) {
    let cur = el;
    while (cur && !(cur.dataset && cur.dataset.model && cur.dataset.tokens)) {
      cur = cur.parentElement;
    }
    return cur || null;
  }

  // --- SLO shape finder (robust to subplot/layout changes) ---
  function getSloShapeIndex(gd) {
    const shapes = (gd.layout && gd.layout.shapes) || [];
    let idx = shapes.findIndex(s => s && s.name === 'slo_line');
    if (idx >= 0) return idx;
    idx = shapes.findIndex(s => s && s.type === 'line' && Number.isFinite(s.x0) && s.x0 === s.x1);
    if (idx >= 0) return idx;
    return 0;
  }

  function getCurrentSlo(gd, fallback = 20) {
    const shapes = (gd.layout && gd.layout.shapes) || [];
    const idx = getSloShapeIndex(gd);
    const s = shapes[idx];
    const slo = s ? Number(s.x0) : fallback;
    return Number.isFinite(slo) ? slo : fallback;
  }

  function getTpsPerRackOverrideFromInteractive(group) {
    if (!group) return null;
    const interactiveDiv = group.querySelector('[id$="-interactive"]');
    if (!interactiveDiv) return null;
    const gd = document.getElementById(interactiveDiv.id);
    if (!gd || !gd._fullLayout || !gd._fullLayout.meta) return null;

    const meta = gd._fullLayout.meta;
    if (!Array.isArray(meta.curveConc) || !meta.curveConc.length) return null;

    const sloIdx0 = getSloShapeIndex(gd);
    Plotly.relayout(gd, { [`shapes[${sloIdx0}].editable`]: false });
    return usersAtSlo(gd, getCurrentSlo(gd, 20));
  }

  function initRackInputsForGroup(group) {
    const rackGraph = group.querySelector('[id$="-rack"]');
    if (!rackGraph) return false;

    const gd = document.getElementById(rackGraph.id);
    if (!gd || !gd._fullLayout || !gd._fullLayout.meta || !gd._fullLayout.meta.device_metadata) {
      return false;
    }

    const metadata = gd._fullLayout.meta.device_metadata;
    const MAX_RACK_KW = gd._fullLayout.meta.max_rack_kw || 36000;
    const model = group.dataset.model;
    const tokens = group.dataset.tokens;
    const inputContainer = document.getElementById(`power-inputs-${model}-${tokens}`);

    if (!inputContainer) return false;
    if (inputContainer.querySelector('input[type="number"]')) return true;  // already built

    metadata.forEach((device, idx) => {
      const inputGroup = document.createElement('div');
      inputGroup.className = 'power-input-row';

      const label = document.createElement('label');
      label.className = 'power-input-label';
      label.textContent = `${device.device_name} :`;

      const input = document.createElement('input');
      input.type = 'number';
      input.value = device.initial_power;
      input.min = '1';
      input.step = '100';
      input.className = 'power-input';
      input.dataset.deviceIdx = idx;
      input.dataset.deviceName = device.device_name;

      input.addEventListener('focus', function () {
        this.style.borderColor = '#76D6FF';
        this.style.backgroundColor = '#2a3a3a';
      });
      input.addEventListener('blur', function () {
        this.style.borderColor = '#555';
        this.style.backgroundColor = '#2a2a2a';
      });

      inputGroup.appendChild(label);
      inputGroup.appendChild(input);
      inputContainer.appendChild(inputGroup);

      input.addEventListener('input', function () {
        const override = getTpsPerRackOverrideFromInteractive(group);
        updateRackGraphFromInputs(gd, metadata, MAX_RACK_KW, model, tokens, override);
      });
    });

    const rackInputsDiv = inputContainer.closest('.rack-power-inputs');
    if (rackInputsDiv) rackInputsDiv.style.display = 'block';

    return true;
  }

  function updateRackGraphFromInputs(gd, metadata, MAX_RACK_KW, model, tokens, tpsPerRackOverride) {
    const inputContainer = document.getElementById(`power-inputs-${model}-${tokens}`);
    if (!inputContainer) return;

    const inputs = inputContainer.querySelectorAll('input[type="number"]');
    const newXData = [];
    const newYData = [];

    metadata.forEach((device, idx) => {
      const input = Array.from(inputs).find(inp => parseInt(inp.dataset.deviceIdx) === idx);
      const server_power = input ? (parseFloat(input.value) || device.initial_power) : device.initial_power;

      const defaultTpsPerRack = device.tps_per_dev * (8 / device.num_device);
      const overrideTpsPerRack = Array.isArray(tpsPerRackOverride) ? tpsPerRackOverride[idx] * (8 / device.num_device) : undefined;
      const tps_per_rack = Number.isFinite(overrideTpsPerRack) ? overrideTpsPerRack : defaultTpsPerRack;

      const rack_power = [0];
      const rack_tps = [0];

      let n = 1;
      while (n * server_power <= MAX_RACK_KW) {
        rack_power.push(n * server_power);
        rack_tps.push(n * tps_per_rack);
        n += 1;
      }

      if (rack_tps.length > 1) {
        rack_tps.push(rack_tps[rack_tps.length - 1]);
        rack_power.push(MAX_RACK_KW);
      }

      newXData.push(rack_power);
      newYData.push(rack_tps);
    });

    Plotly.restyle(gd, { x: newXData, y: newYData });
  }

  // ============================================================
  // Lazy chart rendering
  // Charts are emitted as inert JSON specs (see fig_to_lazy_html in report.py) and
  // only handed to Plotly.newPlot once their group is shown. This keeps the initial
  // page load cheap regardless of how many charts the report contains.
  // ============================================================

  function renderLazyPlot(div) {
    if (!div) return Promise.resolve(null);
    if (div.dataset.rendered === '1' || div.dataset.rendering === '1') return Promise.resolve(div);

    const specEl = document.getElementById(div.id + '-spec');
    if (!specEl) return Promise.resolve(div);

    let spec;
    try {
      spec = JSON.parse(specEl.textContent);
    } catch (e) {
      console.error('Failed to parse plot spec for', div.id, e);
      return Promise.resolve(div);
    }

    div.dataset.rendering = '1';
    return Plotly.newPlot(div, spec.data || [], spec.layout || {}, spec.config || {})
      .then(() => {
        div.dataset.rendered = '1';
        div.dataset.rendering = '';
        if (specEl.parentNode) specEl.parentNode.removeChild(specEl);  // free the JSON payload
        return div;
      })
      .catch((e) => {
        console.error('Plotly.newPlot failed for', div.id, e);
        div.dataset.rendering = '';
        return div;
      });
  }

  function renderGroupPlots(group) {
    const divs = Array.from(group.querySelectorAll('.lazy-plot'));
    return Promise.all(divs.map(renderLazyPlot));
  }

  // ============================================================
  // Interactive plot: SLO line drag/click -> update bar + rack plot
  // ============================================================

  function setupInteractive(gd) {
    if (!gd || gd.__interactiveSetup) return true;
    if (!gd._fullLayout || !gd._fullLayout.meta || typeof gd.on !== 'function') return false;

    const meta = gd._fullLayout.meta;
    if (!Array.isArray(meta.curveConc) || !meta.curveConc.length) return false;

    gd.__interactiveSetup = true;

    const SLO_STEP = 5, SLO_MIN = 20, SLO_MAX = 100;
    let isInternalUpdate = false;
    let updateTimeout = null;

    const snapStep = (v, step) => Math.round(v / step) * step;
    const hasShapeChange = (ev) => ev && Object.keys(ev).some(k => k.startsWith('shapes['));
    const getShapeIndexFromEvent = (ev) => {
      const key = Object.keys(ev).find(k => k.startsWith('shapes['));
      const match = key && key.match(/shapes\[(\d+)\]/);
      return match ? Number(match[1]) : 0;
    };

    function updateBarTraces(newYs) {
      const barIndices = [];
      for (let i = 0; i < gd.data.length; i++) {
        if (gd.data[i].type === 'bar') barIndices.push(i);
      }
      if (!barIndices.length) return Promise.resolve();

      if (barIndices.length === newYs.length) {
        return Plotly.restyle(gd, { y: newYs.map(v => [v]) }, barIndices);
      }

      const labels = meta.labels || [];
      const mappedIndices = [];
      const mappedYs = [];
      barIndices.forEach((ti) => {
        const tr = gd.data[ti];
        const key = tr.legendgroup || tr.name || (Array.isArray(tr.x) ? tr.x[0] : null);
        const j = labels.indexOf(key);
        if (j >= 0) { mappedIndices.push(ti); mappedYs.push([newYs[j]]); }
      });
      if (!mappedIndices.length) return Promise.resolve();
      return Plotly.restyle(gd, { y: mappedYs }, mappedIndices);
    }

    function updateFromShape(ev, useDebounce) {
      if (isInternalUpdate) return;
      if (!hasShapeChange(ev)) return;

      const sloIdx = getSloShapeIndex(gd);
      if (getShapeIndexFromEvent(ev) !== sloIdx) return;

      const layout = gd.layout;
      const keyX0 = `shapes[${sloIdx}].x0`;
      const keyX1 = `shapes[${sloIdx}].x1`;
      const x0 = ev[keyX0] ?? (layout.shapes && layout.shapes[sloIdx] ? layout.shapes[sloIdx].x0 : undefined);
      const x1 = ev[keyX1] ?? (layout.shapes && layout.shapes[sloIdx] ? layout.shapes[sloIdx].x1 : undefined);
      if (!Number.isFinite(x0) || !Number.isFinite(x1)) return;

      let slo = snapStep((x0 + x1) / 2, SLO_STEP);
      slo = Math.max(SLO_MIN, Math.min(SLO_MAX, slo));

      const doUpdate = () => {
        isInternalUpdate = true;
        Plotly.relayout(gd, {
          [keyX0]: slo,
          [keyX1]: slo,
          [`shapes[${sloIdx}].line.color`]: 'white',
          [`shapes[${sloIdx}].line.dash`]: 'dash',
          [`shapes[${sloIdx}].line.width`]: 3,
        }).then(() => {
          const newYs = usersAtSlo(gd, slo);
          return updateBarTraces(newYs).then(() => {
            const group = findParentGroup(gd);
            if (!group) return;
            const rackDiv = group.querySelector('[id$="-rack"]');
            if (!rackDiv) return;
            const rackGd = document.getElementById(rackDiv.id);
            if (!rackGd || !rackGd._fullLayout || !rackGd._fullLayout.meta || !rackGd._fullLayout.meta.device_metadata) return;
            const metadata = rackGd._fullLayout.meta.device_metadata;
            const MAX_RACK_KW = rackGd._fullLayout.meta.max_rack_kw || 36000;
            updateRackGraphFromInputs(rackGd, metadata, MAX_RACK_KW, group.dataset.model, group.dataset.tokens, newYs);
          });
        }).finally(() => { isInternalUpdate = false; });
      };

      if (useDebounce) {
        if (updateTimeout) clearTimeout(updateTimeout);
        updateTimeout = setTimeout(() => { doUpdate(); updateTimeout = null; }, 50);
      } else {
        if (updateTimeout) { clearTimeout(updateTimeout); updateTimeout = null; }
        doUpdate();
      }
    }

    // click/touch to move the SLO line (capture phase so Plotly can't swallow it)
    function xrefToAxisObj(gd, xref) {
      const fl = gd._fullLayout;
      const m = (xref || 'x').match(/^x(\d+)?$/);
      const n = m && m[1] ? m[1] : '';
      const axisKey = `xaxis${n}`;
      return fl && fl[axisKey] ? fl[axisKey] : (fl ? fl.xaxis : null);
    }
    function getClientX(e) {
      if (e.touches && e.touches[0]) return e.touches[0].clientX;
      if (e.changedTouches && e.changedTouches[0]) return e.changedTouches[0].clientX;
      return e.clientX;
    }
    const moveFromEvent = (e, debounce) => {
      if (!gd || !gd._fullLayout) return;
      const sloIdx = getSloShapeIndex(gd);
      const shape = gd.layout && gd.layout.shapes && gd.layout.shapes[sloIdx];
      if (!shape) return;
      const xa = xrefToAxisObj(gd, shape.xref || 'x');
      if (!xa || !Number.isFinite(xa._offset) || !Number.isFinite(xa._length)) return;
      const rect = gd.getBoundingClientRect();
      const xPxInDiv = getClientX(e) - rect.left;
      if (xPxInDiv < xa._offset || xPxInDiv > xa._offset + xa._length) return;
      const xVal = xa.p2l(xPxInDiv - xa._offset);
      if (!Number.isFinite(xVal)) return;
      updateFromShape({ [`shapes[${sloIdx}].x0`]: xVal, [`shapes[${sloIdx}].x1`]: xVal }, debounce);
    };
    gd.addEventListener('pointerdown', (e) => moveFromEvent(e, false), true);
    gd.addEventListener('pointermove', (e) => { if (e.buttons) moveFromEvent(e, true); }, true);

    gd.on('plotly_relayouting', ev => updateFromShape(ev, true));
    gd.on('plotly_relayout', ev => updateFromShape(ev, false));
    return true;
  }

  // ============================================================
  // Graph visibility controller
  // Shows exactly one graph-container (model + task + tokens) and renders its
  // charts lazily, then wires up the interactive + rack behaviours.
  // ============================================================

  function onGroupShown(group) {
    renderGroupPlots(group).then(() => {
      group.querySelectorAll('.js-plotly-plot').forEach((div) => {
        try { Plotly.Plots.resize(div); } catch (_) {}
      });

      const interactiveDiv = group.querySelector('[id$="-interactive"]');
      if (interactiveDiv) setupInteractive(document.getElementById(interactiveDiv.id));

      initRackInputsForGroup(group);
      const rackInputs = group.querySelector('.rack-power-inputs');
      if (rackInputs) rackInputs.style.display = 'block';
    });
  }

  function currentTokenSelect() {
    const modelSelect = document.getElementById('model-select');
    const taskSelect = document.getElementById('task-select');
    if (!modelSelect || !taskSelect) return null;
    return document.getElementById(`token-select-${modelSelect.value}-${taskSelect.value}`);
  }

  function showSelectedGraph() {
    const modelSelect = document.getElementById('model-select');
    const taskSelect = document.getElementById('task-select');
    if (!modelSelect || !taskSelect) return;

    const model = modelSelect.value;
    const task = taskSelect.value;
    const tokenSelect = currentTokenSelect();
    const tokens = tokenSelect ? tokenSelect.value : null;

    // "Overview" is a standalone initial page: show only the summary section and
    // hide every model detail view.
    const isOverview = (model === '__overview__');
    document.querySelectorAll('.summary-section').forEach((sec) => {
      sec.style.display = isOverview ? '' : 'none';
    });

    document.querySelectorAll('.model-section').forEach((section) => {
      section.style.display = (!isOverview && section.dataset.model === model) ? '' : 'none';
    });

    if (isOverview) {
      document.querySelectorAll('.graph-container').forEach((group) => { group.style.display = 'none'; });
      renderSummaryCharts();
      return;
    }

    document.querySelectorAll('.graph-container').forEach((group) => {
      const match = group.dataset.model === model
        && group.dataset.task === task
        && group.dataset.tokens === tokens;
      group.style.display = match ? 'block' : 'none';
      if (match) onGroupShown(group);
    });
  }

  // Landing summary charts live outside the model navigation, so render them
  // eagerly on load and keep them sized to their container.
  function renderSummaryCharts() {
    const sections = document.querySelectorAll('.summary-section');
    sections.forEach((section) => {
      Array.from(section.querySelectorAll('.lazy-plot')).forEach((div) => {
        renderLazyPlot(div).then((el) => {
          if (el) { try { Plotly.Plots.resize(el); } catch (_) {} }
        });
      });
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    const modelSelect = document.getElementById('model-select');
    const taskSelect = document.getElementById('task-select');

    if (modelSelect) modelSelect.addEventListener('change', showSelectedGraph);
    if (taskSelect) taskSelect.addEventListener('change', showSelectedGraph);
    document.querySelectorAll('.token-select').forEach((sel) => {
      sel.addEventListener('change', showSelectedGraph);
    });

    renderSummaryCharts();
    showSelectedGraph();
  });

  // ============================================================
  // ZIP download by selected model
  // ============================================================

  document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById('download-zip-btn');
    const modelSelect = document.getElementById('model-select');
    if (!btn || !modelSelect) return;

    function toFileSafe(s) {
      return String(s ?? '').trim().replaceAll('/', '-').replace(/\s+/g, '_');
    }
    function buildZipUrl(model) {
      return `/csv/${toFileSafe(model)}_offline.zip`;
    }
    function updateButtonState() {
      const model = modelSelect.value;
      btn.disabled = !model;
      btn.dataset.zipUrl = model ? buildZipUrl(model) : '';
    }

    btn.addEventListener('click', function () {
      const url = btn.dataset.zipUrl;
      if (url) window.location.href = url;
    });
    modelSelect.addEventListener('change', updateButtonState);
    updateButtonState();
  });
})();
