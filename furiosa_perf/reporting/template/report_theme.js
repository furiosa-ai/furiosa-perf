/* plotly-theme-manager.js
 *
 * Features:
 * - go.Table theming (header/cells/lines)
 * - Line charts (scatter/scattergl with mode contains "lines") palette recolor (optional)
 * - Layout theming: background, axes, grid, font, legend, hoverlabel, annotations
 * - Shapes line recolor
 * - Works with hidden containers: show first -> resize -> apply theme
 *
 * Page integration:
 * - theme toggle button id="theme-toggle-btn"
 * - model/task select: #model-select, #task-select
 * - token select: multiple .token-select[data-model][data-task] or single #token-select
 * - graph containers: .graph-container[data-model][data-task][data-tokens]
 */

(function (global) {
  const DEFAULT_THEMES = {
    dark: {
      paperBg: "#000000",
      plotBg: "#000000",
      fontColor: "#ffffff",
      gridColor: "rgba(255,255,255,0.12)",
      axisLineColor: "rgba(255,255,255,0.30)",
      zeroLineColor: "rgba(255,255,255,0.18)",
      legendBg: "rgba(0,0,0,0)",

      hoverBg: "#000000",          // (FIX) 누락되어 undefined 들어가던 것 방지

      tableHeaderFill: "#000000",
      tableRowOdd: "#121212",
      tableRowEven: "#000000",     // (FIX) "##000000" 오타
      tableCellLine: "rgba(255,255,255,0.25)",
      tableHeaderLine: "rgba(255,255,255,0.35)",
      tableShapeLine: "rgba(255,255,255,0.35)",

      sloLineColor: "#ffffff",

      // (optional) 패널 테마용
      panelFill: "rgba(20,20,20,0.65)",
      panelBorder: "rgba(255,255,255,0.15)",
      shapeLine: "rgba(255,255,255,0.35)",
    },

    light: {
      paperBg: "#ffffff",
      plotBg: "#ffffff",
      fontColor: "#111111",
      gridColor: "rgba(0,0,0,0.10)",
      axisLineColor: "rgba(0,0,0,0.25)",
      zeroLineColor: "rgba(0,0,0,0.14)",
      legendBg: "rgba(255,255,255,0)",
      hoverBg: "#ffffff",

      tableHeaderFill: "#f2f2f2",
      tableRowOdd: "#ffffff",
      tableRowEven: "#f7f7f7",
      tableCellLine: "rgba(0,0,0,0.15)",
      tableHeaderLine: "rgba(0,0,0,0.22)",
      tableShapeLine: "rgba(0,0,0,0.22)",

      sloLineColor: "#000000",

      // (optional) 패널 테마용
      panelFill: "rgba(20,20,20,0.1)",
      panelBorder: "rgba(255,255,255,0.15)",
      shapeLine: "rgba(255,255,255,0.35)",
    }
  };

  function isPlotlyReady() {
    return typeof global.Plotly !== "undefined";
  }

  function traceModeHasLines(tr) {
    const m = tr && tr.mode ? String(tr.mode) : "";
    return m.includes("lines");
  }

  function getTableTraceIndices(gd) {
    const out = [];
    (gd.data || []).forEach((tr, i) => {
      if (tr && tr.type === "table") out.push(i);
    });
    return out;
  }

  function getLineTraceIndices(gd) {
    const out = [];
    (gd.data || []).forEach((tr, i) => {
      const t = tr?.type;
      const isScatter = (t === "scatter" || t === "scattergl");
      if (isScatter && traceModeHasLines(tr)) out.push(i);
    });
    return out;
  }

  function getUsedAxisKeys(gd) {
    const used = new Set();

    // 기본 축은 항상 사용 가능성 있으니 포함
    used.add("xaxis");
    used.add("yaxis");

    (gd.data || []).forEach((tr) => {
      if (!tr || tr.visible === false || tr.visible === "legendonly") return;

      // trace.xaxis: 'x', 'x2', 'x3'... / yaxis도 동일
      const xa = tr.xaxis ? `xaxis${String(tr.xaxis).slice(1)}` : "xaxis";
      const ya = tr.yaxis ? `yaxis${String(tr.yaxis).slice(1)}` : "yaxis";
      used.add(xa);
      used.add(ya);
    });

    return used;
  }

  function themedLayoutUpdate(gd, theme) {
    const layout = gd.layout || {};

    const layoutUpdate = {
      paper_bgcolor: theme.paperBg,
      plot_bgcolor: theme.plotBg,

      // font도 기존 size/family 유지
      font: {
        ...(layout.font || {}),
        color: theme.fontColor
      },

      hoverlabel: {
        ...(layout.hoverlabel || {}),
        bgcolor: theme.hoverBg ?? (layout.hoverlabel?.bgcolor),
        font: { ...(layout.hoverlabel?.font || {}), color: theme.fontColor }
      },

      // (FIX 핵심) legend 위치(x,y 등) 보존 + 색만 변경
      legend: {
        ...(layout.legend || {}),
        bgcolor: theme.legendBg,
        font: { ...(layout.legend?.font || {}), color: theme.fontColor }
      }
    };

    // 축들만 업데이트: "사용중이면 테마 적용", "미사용이면 숨김"
    Object.keys(layout).forEach((k) => {
      const isXAxis = /^xaxis(\d+)?$/.test(k);
      const isYAxis = /^yaxis(\d+)?$/.test(k);
      if (!isXAxis && !isYAxis) return;

      layoutUpdate[k] = {
        ...(layout[k] || {}),
        gridcolor: theme.gridColor,
        zerolinecolor: theme.zeroLineColor,
        linecolor: theme.axisLineColor,
        tickfont: { ...((layout[k] || {}).tickfont || {}), color: theme.fontColor },
        titlefont: { ...((layout[k] || {}).titlefont || {}), color: theme.fontColor }
      };
    });

    // annotations/font
    if (Array.isArray(layout.annotations)) {
      layoutUpdate.annotations = layout.annotations.map(a => ({
        ...a,
        font: { ...(a.font || {}), color: theme.fontColor }
      }));
    }

    // shapes/line
    const shapes = Array.isArray(layout.shapes) ? layout.shapes : [];

    layoutUpdate.shapes = shapes.map(s => {
      const shapeId = s?.name || s?.templateitemname || s?.label?.text || "";
      const isSlo = shapeId === "slo_line";
      const isPanel = shapeId === "legend_panel";

      const next = { ...s };

      if (next.line) {
        next.line = {
          ...(next.line || {}),
          color: isSlo
            ? (theme.sloLineColor || theme.fontColor)
            : (isPanel ? (theme.panelBorder || next.line.color) : (theme.tableShapeLine || next.line.color))
        };
      }

      if (isPanel) {
        // (중요) 패널은 fill을 테마에서 강제
        if (theme.panelFill) next.fillcolor = theme.panelFill;
      }

      return next;
    });




    return layoutUpdate;
  }

  async function applyToGd(gd, themeName, options) {
    if (!isPlotlyReady()) return; // Plotly 로딩 전이면 조용히 스킵

    const opts = Object.assign(
      { themes: DEFAULT_THEMES, recolorLines: true, recolorTable: true },
      options || {}
    );

    const themes = opts.themes || DEFAULT_THEMES;
    const theme = themes[themeName];
    if (!theme) throw new Error(`Unknown theme: ${themeName}`);

    if (!gd) return;

    // 아직 plot이 생성되기 전이면 다음 tick 재시도
    if (!gd.data || !gd.data.length) {
      setTimeout(() => applyToGd(gd, themeName, opts), 0);
      return;
    }

    // (중요) 숨김 -> 표시 직후에는 크기가 0일 수 있어 resize를 먼저 시도
    try { global.Plotly.Plots.resize(gd); } catch (_) {}

    // 1) TABLE traces
    if (opts.recolorTable) {
      const tableIdxs = getTableTraceIndices(gd);

      for (const idx of tableIdxs) {
        const tr = gd.data[idx];
        const values = tr?.cells?.values || [];
        const nCols = values.length;
        const nRows = Array.isArray(values[0]) ? values[0].length : 0;

        const rowStripe = Array.from({ length: nRows }, (_, r) =>
          (r % 2 === 0) ? theme.tableRowOdd : theme.tableRowEven
        );

        const fill2d = Array.from({ length: nCols }, () => rowStripe);

        await Plotly.restyle(
          gd,
          {
            "header.fill.color": [theme.tableHeaderFill],
            "header.font.color": [theme.fontColor],
            "header.line.color": [theme.tableHeaderLine],

            // 중요: 2D 배열을 trace 1개에 적용하므로 한 번 더 []로 감쌈
            "cells.fill.color": [fill2d],
            "cells.font.color": [theme.fontColor],
            "cells.line.color": [theme.tableCellLine]
          },
          [idx]
        );
      }

    }

    function themedLayoutUpdateSafe(gd, theme) {
      const layout = gd.layout || {};
      const upd = {
        "paper_bgcolor": theme.paperBg,
        "plot_bgcolor": theme.plotBg,
        "font.color": theme.fontColor,
        "legend.bgcolor": theme.legendBg,
        "legend.font.color": theme.fontColor,
        "hoverlabel.bgcolor": theme.hoverBg,
        "hoverlabel.font.color": theme.fontColor,

        // 중요: UI 상태 유지
        "uirevision": "theme-lock",
      };

      Object.keys(layout).forEach((k) => {
        if (!/^xaxis(\d+)?$/.test(k) && !/^yaxis(\d+)?$/.test(k)) return;

        upd[`${k}.gridcolor`] = theme.gridColor;
        upd[`${k}.zerolinecolor`] = theme.zeroLineColor;
        upd[`${k}.linecolor`] = theme.axisLineColor;

        // tick/title은 "색상만" 바꾸기
        upd[`${k}.tickfont.color`] = theme.fontColor;

        // 최신 plotly에선 titlefont 대신 title.font 권장
        upd[`${k}.title.font.color`] = theme.fontColor;
      });

      return upd;
    }

    // 3) LAYOUT
    const layoutUpdate = themedLayoutUpdateSafe(gd, theme);
    layoutUpdate.uirevision = "theme-lock"; // 아무 고정 문자열이면 됨
    await global.Plotly.relayout(gd, layoutUpdate);

    gd.dataset.plotlyTheme = themeName;
  }

  async function apply(divId, themeName, options) {
    const gd = document.getElementById(divId);
    return applyToGd(gd, themeName, options);
  }

  function applyThemeToContainer(containerEl, themeName, options) {
    if (!containerEl) return;
    const plots = containerEl.querySelectorAll(".js-plotly-plot, .plotly-graph-div");
    plots.forEach((gd) => {
      // show 직후 1 tick 뒤에 resize + apply를 하는 게 안정적
      requestAnimationFrame(() => {
        try { global.Plotly?.Plots?.resize(gd); } catch (_) {}
        applyToGd(gd, themeName, options);
      });
    });
  }

  // ----- Page integration (show/hide + theme) -----
  document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    const btn = document.getElementById("theme-toggle-btn");

    const modelSelect = document.getElementById("model-select");
    const taskSelect = document.getElementById("task-select");

    const esc = (s) => {
      if (global.CSS && typeof global.CSS.escape === "function") return global.CSS.escape(String(s));
      return String(s).replace(/["\\]/g, "\\$&");
    };

    function getCurrentTokenSelect() {
      const m = modelSelect?.value || "";
      const t = taskSelect?.value || "";

      const sel = document.querySelector(
        `.token-select[data-model="${esc(m)}"][data-task="${esc(t)}"]`
      );
      if (sel) return sel;

      return document.getElementById("token-select");
    }

    function getSelected() {
      const tokenSel = getCurrentTokenSelect();
      return {
        model: modelSelect?.value || "",
        task: taskSelect?.value || "",
        tokens: tokenSel?.value || ""
      };
    }

    function getTheme() {
      return root.getAttribute("data-theme") || localStorage.getItem("theme") || "dark";
    }

    function setTheme(theme) {
      root.setAttribute("data-theme", theme);
      localStorage.setItem("theme", theme);
      if (btn) btn.textContent = (theme === "dark") ? "Light mode" : "Dark mode";
    }

    // 선택된 model/task/tokens에 해당하는 컨테이너만 표시
    function updateVisibleContainers(sel) {
      const containers = document.querySelectorAll(".graph-container");
      const visible = [];

      containers.forEach((el) => {
        const match =
          (!sel.model || el.dataset.model === sel.model) &&
          (!sel.task || el.dataset.task === sel.task) &&
          (!sel.tokens || el.dataset.tokens === sel.tokens);

        el.style.display = match ? "block" : "none";
        if (match) visible.push(el);
      });

      return visible;
    }

    // 표시된 컨테이너들에 대해: resize + theme 적용
    function refreshPlotsForSelection() {
      const theme = getTheme();
      setTheme(theme);

      const sel = getSelected();
      const visibleContainers = updateVisibleContainers(sel);

      // 컨테이너가 실제 레이아웃에 반영된 뒤 처리
      requestAnimationFrame(() => {
        visibleContainers.forEach((containerEl) => {
          // (optional) 컨테이너 자체도 한 번 더 next frame에서 안정화
          requestAnimationFrame(() => {
            applyThemeToContainer(containerEl, theme, { recolorLines: true, recolorTable: true });
          });
        });
      });
    }

    // 초기: theme 적용 + 선택값에 맞게 표시 + resize
    const initialTheme = localStorage.getItem("theme") || root.getAttribute("data-theme") || "dark";
    setTheme(initialTheme);
    refreshPlotsForSelection();

    // theme toggle
    if (btn) {
      btn.addEventListener("click", () => {
        const cur = getTheme();
        const next = (cur === "dark") ? "light" : "dark";
        setTheme(next);
        refreshPlotsForSelection();
      });
    }

    // selection changes
    if (modelSelect) modelSelect.addEventListener("change", refreshPlotsForSelection);
    if (taskSelect) taskSelect.addEventListener("change", refreshPlotsForSelection);

    // token select는 여러 개일 수 있음
    document.addEventListener("change", (e) => {
      const target = e.target;
      if (!target) return;
      if (target.classList?.contains("token-select")) refreshPlotsForSelection();
      if (target.id === "token-select") refreshPlotsForSelection();
    });

    // (추가 안전장치) Plotly가 늦게 생성되는 경우를 위해 잠깐 후 한 번 더
    setTimeout(refreshPlotsForSelection, 50);
    setTimeout(refreshPlotsForSelection, 250);
  });

  global.PlotlyThemeManager = {
    apply,
    applyToGd,
    applyThemeToContainer,
    DEFAULT_THEMES
  };
})(window);
