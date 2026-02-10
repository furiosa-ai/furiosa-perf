(() => {
  // Shared helper (used by both the interactive plot and rack plot override)
  function computeMaxConcurrency(a, b, slo) {
    if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0 || b <= 0 || slo <= 0) return 0;
    const conc = a / slo - b;
    return Number.isFinite(conc) ? Math.max(0, conc) : 0;
  }

  function findParentGroup(el) {
    let cur = el;
    while (cur && !(cur.dataset && cur.dataset.model && cur.dataset.tokens)) {
      cur = cur.parentElement;
    }
    return cur || null;
  }

  // --- NEW: SLO shape finder (robust to subplot/layout changes) ---
  function getSloShapeIndex(gd) {
    const shapes = (gd.layout && gd.layout.shapes) || [];

    // 1) Prefer explicit name
    let idx = shapes.findIndex(s => s && s.name === 'slo_line');
    if (idx >= 0) return idx;

    // 2) Fallback: find a vertical line shape
    idx = shapes.findIndex(s => s && s.type === 'line' && Number.isFinite(s.x0) && s.x0 === s.x1);
    if (idx >= 0) return idx;

    // 3) Last resort
    return 0;
  }

  function getCurrentSlo(gd, fallback = 50) {
    const shapes = (gd.layout && gd.layout.shapes) || [];
    const idx = getSloShapeIndex(gd);
    const s = shapes[idx];
    const slo = s ? Number(s.x0) : fallback;
    return Number.isFinite(slo) ? slo : fallback;
  }

  function getTpsPerRackOverrideFromInteractive(group) {
    if (!group) return null;
    const interactiveDiv = group.querySelector('[id$="-offline-interactive"]');
    if (!interactiveDiv) return null;
    const gd = document.getElementById(interactiveDiv.id);
    if (!gd || !gd._fullLayout || !gd._fullLayout.meta) return null;

    const fitA = gd._fullLayout.meta.fitA || [];
    const fitB = gd._fullLayout.meta.fitB || [];
    if (!fitA.length || !fitB.length) return null;

    const sloIdx0 = getSloShapeIndex(gd);
    Plotly.relayout(gd, {
      [`shapes[${sloIdx0}].editable`]: false,   // 핵심: 이 shape는 드래그 편집 금지
    });
    // CHANGED: do not assume shapes[0]
    const slo = getCurrentSlo(gd, 50);
    return fitA.map((A, i) => computeMaxConcurrency(A, fitB[i], slo));
  }

  function initRackInputsForGroup(group) {
    const rackGraph = group.querySelector('[id$="-offline-rack"]');
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

    if (inputContainer.querySelector('input[type="number"]')) {
      return true;
    }

    metadata.forEach((device, idx) => {
      const inputGroup = document.createElement('div');
      // one row per device: "Device : [input]"
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

      // default tps_per_rack from metadata, but allow override from interactive plot (newYs)
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

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".offline-token-select").forEach(function (select) {
      const model = select.dataset.model;
      const groups = document.querySelectorAll('.graph-container[data-model="' + model + '"]');

      function resizePlotsIn(group) {
        group.querySelectorAll('.js-plotly-plot').forEach(function (div) {
          if (window.Plotly && Plotly.Plots && Plotly.Plots.resize) {
            Plotly.Plots.resize(div);
          }
        });
      }

      function updateOfflineView() {
        const selected = select.value;
        groups.forEach(function (group) {
          if (group.dataset.tokens === selected) {
            group.style.display = "block";
            resizePlotsIn(group);

            const rackInputs = group.querySelector('.rack-power-inputs');
            if (rackInputs) rackInputs.style.display = "block";

            let attempts = 0;
            const tryInit = setInterval(function () {
              attempts++;
              if (initRackInputsForGroup(group) || attempts >= 20) clearInterval(tryInit);
            }, 200);
          } else {
            group.style.display = "none";
            const rackInputs = group.querySelector('.rack-power-inputs');
            if (rackInputs) rackInputs.style.display = "none";
          }
        });
      }

      select.addEventListener("change", updateOfflineView);

      if (select.options.length > 0) {
        select.selectedIndex = 0;
        updateOfflineView();

        setTimeout(function () {
          groups.forEach(function (group) {
            if (group.dataset.tokens === select.value) {
              let attempts = 0;
              const tryInit = setInterval(function () {
                attempts++;
                if (initRackInputsForGroup(group) || attempts >= 20) clearInterval(tryInit);
              }, 200);
            }
          });
        }, 1000);
      }
    });
  });

  document.addEventListener('DOMContentLoaded', function () {
    const select = document.getElementById('model-select');
    const sections = document.querySelectorAll('.model-section');
    const offlineFields = document.querySelectorAll('.offline-field');

    function filterSections() {
      const selected = select.value;
      sections.forEach(section => {
        const model = section.dataset.model;
        section.style.display = (model === selected) ? '' : 'none';
      });

      // show the selected model's offline-token-select field (top controls)
      offlineFields.forEach(field => {
        const model = field.dataset.model;
        field.style.display = (model === selected) ? '' : 'none';
      });

      // re-run the selected model's offline view logic to ensure Plotly resizes correctly
      const offlineSelect = document.getElementById(`offline-token-select-${selected}`);
      if (offlineSelect) {
        offlineSelect.dispatchEvent(new Event('change'));
      }
    }

    filterSections();
    select.addEventListener('change', filterSections);
  });

  document.addEventListener('DOMContentLoaded', function () {
    function initAllRackInputs() {
      const rackGraphDivs = document.querySelectorAll('[id$="-offline-rack"]');

      rackGraphDivs.forEach((gd) => {
        let parentContainer = gd.parentElement;
        while (parentContainer && !parentContainer.dataset.model) {
          parentContainer = parentContainer.parentElement;
        }
        if (!parentContainer || !parentContainer.dataset.model) return;

        initRackInputsForGroup(parentContainer);
      });
    }

    let attempts = 0;
    const maxAttempts = 50;
    const initInterval = setInterval(function () {
      attempts++;
      initAllRackInputs();
      if (attempts >= maxAttempts) clearInterval(initInterval);
    }, 200);

    document.addEventListener('plotly_afterplot', function () {
      initAllRackInputs();
    });
  });

  // ====== Interactive plot: SLO line drag -> update bar + rack plot ======
  document.addEventListener('DOMContentLoaded', function () {
    const graphDivs = document.querySelectorAll('[id$="-offline-interactive"]');

    graphDivs.forEach((gd) => {
      let isInternalUpdate = false;

      function tryAttach() {
        if (!gd || !gd._fullLayout || !gd._fullLayout.meta || typeof gd.on !== 'function') {
          return false;
        }

        const fitA = gd._fullLayout.meta.fitA || [];
        const fitB = gd._fullLayout.meta.fitB || [];
        if (!fitA.length || !fitB.length) return false;

        const labels = gd._fullLayout.meta.labels || []; // devices list (python meta.labels)

        const SLO_STEP = 5, SLO_MIN = 20, SLO_MAX = 100;
        let updateTimeout = null;

        const snapStep = (v, step) => Math.round(v / step) * step;

        const hasShapeChange = (ev) =>
          ev && Object.keys(ev).some(key => key.startsWith('shapes['));

        const getShapeIndexFromEvent = (ev) => {
          const key = Object.keys(ev).find(k => k.startsWith('shapes['));
          if (!key) return 0;
          const match = key.match(/shapes\[(\d+)\]/);
          return match ? Number(match[1]) : 0;
        };

        // CHANGED: bar 업데이트는 축(x2/y2 등) 가정 제거
        function updateBarTraces(newYs) {
          // 1) bar trace 인덱스 전부 수집 (축 번호에 의존하지 않음)
          const barIndices = [];
          for (let i = 0; i < gd.data.length; i++) {
            const tr = gd.data[i];
            if (tr.type === 'bar') barIndices.push(i);
          }

          if (!barIndices.length) return Promise.resolve();

          // 2) "장치별 bar trace" 개수 == newYs(장치 수) 이면 순서대로 업데이트
          if (barIndices.length === newYs.length) {
            const ys = newYs.map(v => [v]); // 각 bar trace는 y=[value]
            return Plotly.restyle(gd, { y: ys }, barIndices);
          }

          // 3) fallback: meta.labels와 legendgroup/name/x[0]로 매핑해서 업데이트
          const labels = (gd._fullLayout && gd._fullLayout.meta && gd._fullLayout.meta.labels) || [];
          const mappedIndices = [];
          const mappedYs = [];

          barIndices.forEach((ti) => {
            const tr = gd.data[ti];
            const key =
              tr.legendgroup ||
              tr.name ||
              (Array.isArray(tr.x) ? tr.x[0] : null);

            const j = labels.indexOf(key);
            if (j >= 0) {
              mappedIndices.push(ti);
              mappedYs.push([newYs[j]]);
            }
          });

          if (!mappedIndices.length) return Promise.resolve();
          return Plotly.restyle(gd, { y: mappedYs }, mappedIndices);
        }

        function updateFromShape(ev, useDebounce) {
          if (isInternalUpdate) return;
          if (!hasShapeChange(ev)) return;

          // CHANGED: only respond to SLO line shape updates
          const sloIdx = getSloShapeIndex(gd);
          const changedIdx = getShapeIndexFromEvent(ev);
          if (changedIdx !== sloIdx) return;

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
              const newYs = fitA.map((A, i) => computeMaxConcurrency(A, fitB[i], slo));

              return updateBarTraces(newYs).then(() => {
                // also refresh rack graph in the same (model,tokens) group using newYs as tps_per_rack override
                const group = findParentGroup(gd);
                if (!group) return;

                const rackDiv = group.querySelector('[id$="-offline-rack"]');
                if (!rackDiv) return;

                const rackGd = document.getElementById(rackDiv.id);
                if (!rackGd || !rackGd._fullLayout || !rackGd._fullLayout.meta || !rackGd._fullLayout.meta.device_metadata) return;

                const metadata = rackGd._fullLayout.meta.device_metadata;
                const MAX_RACK_KW = rackGd._fullLayout.meta.max_rack_kw || 36000;
                const model = group.dataset.model;
                const tokens = group.dataset.tokens;

                updateRackGraphFromInputs(rackGd, metadata, MAX_RACK_KW, model, tokens, newYs);
              });
            }).finally(() => {
              isInternalUpdate = false;
            });
          };

          if (useDebounce) {
            if (updateTimeout) clearTimeout(updateTimeout);
            updateTimeout = setTimeout(() => {
              doUpdate();
              updateTimeout = null;
            }, 50);
          } else {
            if (updateTimeout) {
              clearTimeout(updateTimeout);
              updateTimeout = null;
            }
            doUpdate();
          }
        }
        // ===== REPLACE WITH THIS: click/touch to move SLO line (capture phase) =====
        function xrefToAxisObj(gd, xref) {
          const fl = gd._fullLayout;
          const m = (xref || 'x').match(/^x(\d+)?$/);
          const n = m && m[1] ? m[1] : '';
          const axisKey = `xaxis${n}`;
          return fl && fl[axisKey] ? fl[axisKey] : (fl ? fl.xaxis : null);
        }

        function getClientX(e) {
          // pointer/mouse/touch 모두 처리
          if (e.touches && e.touches[0]) return e.touches[0].clientX;
          if (e.changedTouches && e.changedTouches[0]) return e.changedTouches[0].clientX;
          return e.clientX;
        }

        function attachClickToMoveSlo() {
          if (gd.__sloClickAttached) return;
          gd.__sloClickAttached = true;

          const moveFromEvent = (e, debounce) => {
            if (!gd || !gd._fullLayout) return;

            const sloIdx = getSloShapeIndex(gd);
            const shape = gd.layout?.shapes?.[sloIdx];
            if (!shape) return;

            const xa = xrefToAxisObj(gd, shape.xref || 'x');
            if (!xa || !Number.isFinite(xa._offset) || !Number.isFinite(xa._length)) return;

            const rect = gd.getBoundingClientRect();
            const clientX = getClientX(e);
            const xPxInDiv = clientX - rect.left;

            const left = xa._offset;
            const right = xa._offset + xa._length;

            // 플롯 영역 밖이면 무시 (원하면 clamp로 바꿀 수 있음)
            if (xPxInDiv < left || xPxInDiv > right) return;

            const xVal = xa.p2l(xPxInDiv - xa._offset);
            if (!Number.isFinite(xVal)) return;

            updateFromShape({
              [`shapes[${sloIdx}].x0`]: xVal,
              [`shapes[${sloIdx}].x1`]: xVal,
            }, debounce);
          };

          // 중요: capture=true 로 달아서 Plotly가 stopPropagation 해도 먼저 잡음
          gd.addEventListener('pointerdown', (e) => moveFromEvent(e, false), true);

          // (옵션) 누르고 드래그로 계속 따라오게
          gd.addEventListener('pointermove', (e) => {
            if (e.buttons) moveFromEvent(e, true);
          }, true);
        }

        attachClickToMoveSlo();
        // ===== END REPLACE =====

        gd.on('plotly_relayouting', ev => updateFromShape(ev, true));
        gd.on('plotly_relayout', ev => updateFromShape(ev, false));
        return true;
      }

      if (!tryAttach()) {
        const interval = setInterval(() => {
          if (tryAttach()) clearInterval(interval);
        }, 100);
      }
    });
  });

    // ====== NEW: ZIP download by (model, tokens) ======
    document.addEventListener('DOMContentLoaded', function () {
      const btn = document.getElementById('download-zip-btn');
      const modelSelect = document.getElementById('model-select');
      if (!btn || !modelSelect) return;

      // tokens 값에 '/', 공백 등 파일명에 애매한 문자가 있으면 치환
      function toFileSafe(s) {
        return String(s ?? '')
          .trim()
          .replaceAll('/', '-')     // 1024/1024 -> 1024-1024
          .replace(/\s+/g, '_');    // 공백 -> _
      }

      function getSelectedModel() {
        return modelSelect.value;
      }

      // ZIP 파일 규칙: /csv/{model}__{tokens}.zip
      function buildZipUrl(model) {
        const m = toFileSafe(model);
        return `/csv/${m}_offline.zip`;
      }

      function updateButtonState() {
        const model = getSelectedModel();
        btn.disabled = !(model);
        btn.dataset.zipUrl = (model) ? buildZipUrl(model) : '';
      }

      // 다운로드 실행
      btn.addEventListener('click', function () {
        const url = btn.dataset.zipUrl;
        if (!url) return;

        // 가장 가벼운 방식: 브라우저가 URL로 직접 다운로드
        window.location.href = url;
      });


      modelSelect.addEventListener('change', function () {
        updateButtonState();
      });

      // 초기 1회
      updateButtonState();
    });
    // ====== END NEW ======
})();
