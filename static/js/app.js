/* ── FitStats Pro — Ghost Busters ── */
/* Frontend logic: tab navigation, API calls, Plotly.js charts */

const PLOTLY_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, sans-serif", color: "#94a3b8", size: 12 },
  margin: { t: 40, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: "rgba(99,102,241,0.08)", zerolinecolor: "rgba(99,102,241,0.15)" },
  yaxis: { gridcolor: "rgba(99,102,241,0.08)", zerolinecolor: "rgba(99,102,241,0.15)" },
};
const PLOTLY_CONFIG = { responsive: true, displayModeBar: false, displaylogo: false };
const COLORS = ["#6366f1","#8b5cf6","#06b6d4","#22c55e","#f59e0b","#ef4444","#ec4899","#14b8a6","#f97316","#a855f7"];

let COLUMNS = { numeric: [], categorical: [] };

// ── Initialization ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await loadColumns();
  setupTabs();
  loadOverview();
});

async function api(url, opts = {}) {
  const r = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  return r.json();
}

// ── Tab Navigation ──────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((t) => t.classList.remove("active"));
      item.classList.add("active");
      const tab = item.dataset.tab;
      document.getElementById("tab-" + tab).classList.add("active");
      if (tab === "correlation") loadHeatmap();
    });
  });
}

// ── Load Columns & Populate Selects ─────────────────────────
async function loadColumns() {
  COLUMNS = await api("/api/columns");
  const nc = COLUMNS.numeric;
  const cc = COLUMNS.categorical;
  const all = COLUMNS.all_analysis || [...nc, ...cc];

  // Helper: find index of a value in an array (for setting defaults)
  const idx = (arr, val) => { const i = arr.indexOf(val); return i >= 0 ? i : 0; };

  // Tab 2: Graphs — defaults chosen for logical relevance to BodyFat prediction
  populateSelect("hist-var", nc, idx(nc, "BodyFat"));              // Target distribution
  populateSelect("scatter-x", nc, idx(nc, "Abdomen"));             // Strongest predictor of BodyFat
  populateSelect("scatter-y", nc, idx(nc, "BodyFat"));             // Target variable
  populateSelect("bar-var", cc, idx(cc, "Gender"));                   // Only meaningful categorical
  populateSelect("pie-var", cc, idx(cc, "Gender"));                   // Only meaningful categorical
  populateSelect("box-var", nc, idx(nc, "BodyFat"));               // Target — check for outliers

  // Tab 3: Descriptive Stats
  populateSelect("desc-var", nc, idx(nc, "BodyFat"));              // Start with the target

  // Tab 4: Distributions
  populateSelect("uniform-var", nc, idx(nc, "Age"));               // Age is closer to uniform spread
  populateSelect("normal-var", nc, idx(nc, "weight"));             // Weight is naturally normal

  // Tab 5: Correlation & Regression
  populateSelect("corr-var1", nc, idx(nc, "Abdomen"));             // Known strong predictor
  populateSelect("corr-var2", nc, idx(nc, "BodyFat"));             // Target
  populateSelect("reg-indep", nc, idx(nc, "Abdomen"));             // Using Abdomen (strongest)

  // Tab 1: Frequency table — all analysis columns (excludes Original)
  populateSelect("freq-var", all, idx(all, "BodyFat"));
}

function populateSelect(id, options, selectedIdx = 0) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = options.map((o, i) => `<option value="${o}" ${i === selectedIdx ? "selected" : ""}>${o}</option>`).join("");
}

// ── Tab 1: Data Overview ────────────────────────────────────
async function loadOverview() {
  const data = await api("/api/preview?n=10");
  const featureData = await api("/api/feature-info");
  // Dataset info
  document.getElementById("dataset-info").innerHTML = `
    <div class="stat-grid">
      <div class="stat-item"><div class="stat-label">Rows</div><div class="stat-value">${data.shape[0]}</div></div>
      <div class="stat-item"><div class="stat-label">Columns</div><div class="stat-value">${data.shape[1]}</div></div>
      <div class="stat-item"><div class="stat-label">Numeric</div><div class="stat-value">${COLUMNS.numeric.length}</div></div>
      <div class="stat-item"><div class="stat-label">Categorical</div><div class="stat-value">2</div></div>
    </div>`;
  // Feature descriptions table
  const features = featureData.features;
  const featureRows = Object.entries(features).map(([col, info]) => {
    const roleColor = info.role === 'Target' ? '#ef4444' : '#22c55e';
    return `<tr><td>${col}</td><td><span style="color:${roleColor};font-weight:600">${info.role}</span></td><td>${info.unit}</td><td>${info.description}</td></tr>`;
  }).join("");
  document.getElementById("feature-descriptions").innerHTML = `
    <table><thead><tr><th>Column</th><th>Role</th><th>Unit</th><th>Description</th></tr></thead>
    <tbody>${featureRows}</tbody></table>
    <div class="interpretation-box" style="margin-top:12px">This dataset contains <strong>${data.shape[0]}</strong> observations with <strong>1 target variable</strong> (BodyFat %), 
    <strong>2 categorical features</strong>, and <strong>13 numeric features</strong> (anthropometric measurements). 
    The goal is to predict body fat percentage using simple, easily obtainable physical measurements.</div>`;
  // Column types
  const analysisColumns = Object.entries(data.dtypes).filter(([col]) => col !== 'Original');
  const typesHTML = analysisColumns.map(([col, dtype]) =>
    `<tr><td>${col}</td><td><span style="color:${dtype.includes("float") || dtype.includes("int") ? "#6366f1" : "#22c55e"}">${dtype}</span></td></tr>`
  ).join("");
  document.getElementById("column-types").innerHTML = `<table><thead><tr><th>Column</th><th>Type</th></tr></thead><tbody>${typesHTML}</tbody></table>`;
  // Data preview table
  const headRow = data.columns.map((c) => `<th>${c}</th>`).join("");
  const bodyRows = data.data.map((row) => `<tr>${row.map((v) => `<td>${v !== null ? v : "N/A"}</td>`).join("")}</tr>`).join("");
  document.getElementById("data-preview").innerHTML = `<table><thead><tr>${headRow}</tr></thead><tbody>${bodyRows}</tbody></table>`;
}

async function loadFrequency() {
  const v = document.getElementById("freq-var").value;
  const bins = parseInt(document.getElementById("freq-bins").value) || 7;
  const data = await api("/api/frequency", { method: "POST", body: { variable: v, bins } });
  if (data.error) { alert(data.error); return; }
  const headRow = "<tr><th>Class</th><th>Freq</th><th>Rel. Freq</th><th>%</th><th>Cum. Freq</th></tr>";
  const bodyRows = data.table.map((r) =>
    `<tr><td>${r.class}</td><td>${r.frequency}</td><td>${r.relative_frequency}</td><td>${r.percentage}%</td><td>${r.cumulative_frequency}</td></tr>`
  ).join("");
  document.getElementById("freq-table-container").innerHTML =
    `<table><thead>${headRow}</thead><tbody>${bodyRows}</tbody></table>
     <div class="interpretation-box mt">${data.interpretation || ''}</div>
     <p style="margin-top:8px;color:var(--text-muted);font-size:12px">Total observations: ${data.total}</p>`;
}

// ── Tab 2: Graphs ───────────────────────────────────────────
async function plotHistogram() {
  const v = document.getElementById("hist-var").value;
  const d = await api("/api/data/column", { method: "POST", body: { variable: v } });
  Plotly.newPlot("hist-chart", [{
    x: d.values, type: "histogram", marker: { color: "#6366f1", line: { color: "#818cf8", width: 1 } },
    opacity: 0.85, nbinsx: 20,
  }], { ...PLOTLY_LAYOUT, title: { text: `Histogram of ${v}`, font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: v }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Frequency" }
  }, PLOTLY_CONFIG);
  showInterp("hist-interp", d.interpretation);
}

async function plotScatter() {
  const x = document.getElementById("scatter-x").value;
  const y = document.getElementById("scatter-y").value;
  const d = await api("/api/data/scatter", { method: "POST", body: { x, y, color: "Gender" } });
  const traces = [];
  if (d.color) {
    const groups = {};
    d.x.forEach((xv, i) => {
      const g = d.color[i];
      if (!groups[g]) groups[g] = { x: [], y: [] };
      groups[g].x.push(xv);
      groups[g].y.push(d.y[i]);
    });
    Object.entries(groups).forEach(([name, vals], i) => {
      traces.push({ x: vals.x, y: vals.y, mode: "markers", type: "scatter", name,
        marker: { color: COLORS[i % COLORS.length], size: 6, opacity: 0.7 } });
    });
  } else {
    traces.push({ x: d.x, y: d.y, mode: "markers", type: "scatter", marker: { color: "#6366f1", size: 6 } });
  }
  Plotly.newPlot("scatter-chart", traces, {
    ...PLOTLY_LAYOUT, title: { text: `${y} vs ${x}`, font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: x }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: y },
    legend: { font: { color: "#94a3b8" } },
  }, PLOTLY_CONFIG);
  showInterp("scatter-interp", d.interpretation);
}

async function plotBar() {
  const v = document.getElementById("bar-var").value;
  const d = await api("/api/data/bar", { method: "POST", body: { variable: v } });
  Plotly.newPlot("bar-chart", [{
    x: d.categories, y: d.counts, type: "bar",
    marker: { color: COLORS.slice(0, d.categories.length), line: { width: 1, color: "rgba(255,255,255,0.1)" } },
  }], { ...PLOTLY_LAYOUT, title: { text: `Bar Chart for ${v}`, font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: "Categories" }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Count" }
  }, PLOTLY_CONFIG);
  showInterp("bar-interp", d.interpretation);
}

async function plotPie() {
  const v = document.getElementById("pie-var").value;
  const d = await api("/api/data/bar", { method: "POST", body: { variable: v } });
  Plotly.newPlot("pie-chart", [{
    labels: d.categories, values: d.counts, type: "pie", hole: 0.4,
    marker: { colors: COLORS.slice(0, d.categories.length) },
    textinfo: "label+percent", textfont: { color: "#f1f5f9" },
  }], { ...PLOTLY_LAYOUT, title: { text: `Pie Chart for ${v}`, font: { size: 16, color: "#f1f5f9" } },
    legend: { font: { color: "#94a3b8" } },
  }, PLOTLY_CONFIG);
  showInterp("pie-interp", d.interpretation);
}

async function plotBox() {
  const v = document.getElementById("box-var").value;
  const d = await api("/api/data/boxplot", { method: "POST", body: { variable: v } });
  Plotly.newPlot("box-chart", [{
    y: d.values, type: "box", name: v, marker: { color: "#6366f1" },
    boxpoints: "outliers", jitter: 0.3, pointpos: -1.8,
  }], { ...PLOTLY_LAYOUT, title: { text: `Box Plot of ${v}`, font: { size: 16, color: "#f1f5f9" } },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: v },
  }, PLOTLY_CONFIG);
  document.getElementById("box-interpretation").innerHTML = d.interpretation;
}

// ── Tab 3: Descriptive Statistics ───────────────────────────
async function loadDescriptive() {
  const v = document.getElementById("desc-var").value;
  const d = await api("/api/descriptive", { method: "POST", body: { variable: v } });
  if (d.error) { alert(d.error); return; }
  const s = d.stats;
  const statItems = [
    ["Count", s.count], ["Mean", s.mean], ["Median", s.median], ["Mode", s.mode],
    ["Std Dev", s.std], ["Variance", s.variance], ["Min", s.min], ["Max", s.max],
    ["Range", s.range], ["Q1", s.q1], ["Q2", s.q2], ["Q3", s.q3],
    ["IQR", s.iqr], ["CV (%)", s.cv], ["Skewness", s.skewness],
  ];
  const grid = statItems.map(([label, val]) =>
    `<div class="stat-item"><div class="stat-label">${label}</div><div class="stat-value">${typeof val === "number" ? val.toFixed(4) : val}</div></div>`
  ).join("");
  const interps = Object.values(d.interpretations).map((t) => `<div class="interpretation-box">${t}</div>`).join("");
  document.getElementById("desc-results").innerHTML = `
    <div class="card"><div class="card-header"><h3>Results for ${v}</h3></div>
      <div class="card-body"><div class="stat-grid">${grid}</div>${interps}</div></div>`;
}

// ── Tab 4: Distributions ────────────────────────────────────
async function plotUniform() {
  const v = document.getElementById("uniform-var").value;
  const d = await api("/api/distribution/uniform", { method: "POST", body: { variable: v } });
  const s = d.stats;
  Plotly.newPlot("uniform-chart", [{
    x: d.samples, type: "histogram", histnorm: "probability density",
    marker: { color: "rgba(99,102,241,0.6)", line: { color: "#818cf8", width: 1 } }, nbinsx: 25, name: "Samples",
  }, {
    x: [s.min, s.min, s.max, s.max], y: [0, s.pdf, s.pdf, 0], mode: "lines",
    line: { color: "#ef4444", width: 3 }, name: "PDF",
  }], { ...PLOTLY_LAYOUT, title: { text: `Uniform Distribution for ${v}`, font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: v, type: 'linear' }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Density" },
    legend: { font: { color: "#94a3b8" } },
  }, PLOTLY_CONFIG);
  document.getElementById("uniform-stats").innerHTML = `<div class="stat-grid">
    ${statBox("Min", s.min)}${statBox("Max", s.max)}${statBox("Mean", s.mean)}
    ${statBox("Variance", s.variance)}${statBox("Std Dev", s.std)}${statBox("PDF", s.pdf)}
  </div>`;
  document.getElementById("uniform-interp").innerHTML = d.interpretation;
}

async function plotNormal() {
  const v = document.getElementById("normal-var").value;
  const d = await api("/api/distribution/normal", { method: "POST", body: { variable: v } });
  const s = d.stats;
  Plotly.newPlot("normal-chart", [{
    x: d.actual_values, type: "histogram", histnorm: "probability density",
    marker: { color: "rgba(99,102,241,0.5)", line: { color: "#818cf8", width: 1 } }, nbinsx: 25, name: "Actual Data",
  }, {
    x: d.pdf_curve.x, y: d.pdf_curve.y, mode: "lines",
    line: { color: "#ef4444", width: 3 }, name: "Normal PDF",
  }], { ...PLOTLY_LAYOUT, title: { text: `Normal Distribution for ${v}`, font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: v, type: 'linear' }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: "Density" },
    legend: { font: { color: "#94a3b8" } },
  }, PLOTLY_CONFIG);
  document.getElementById("normal-stats").innerHTML = `<div class="stat-grid">
    ${statBox("Mean (μ)", s.mean)}${statBox("Std Dev (σ)", s.std)}
    ${statBox("Variance", s.variance)}${statBox("PDF at μ", s.pdf_at_mean)}
  </div>`;
  document.getElementById("normal-interp").innerHTML = d.interpretation;
}

function statBox(label, value) {
  const v = typeof value === "number" ? value.toFixed(4) : value;
  return `<div class="stat-item"><div class="stat-label">${label}</div><div class="stat-value" style="font-size:16px">${v}</div></div>`;
}

function showInterp(id, text) {
  const el = document.getElementById(id);
  if (el && text) { el.innerHTML = text; el.style.display = "block"; }
  else if (el) { el.style.display = "none"; }
}

// ── Tab 5: Correlation & Regression ─────────────────────────
async function loadHeatmap() {
  const d = await api("/api/correlation/matrix");
  Plotly.newPlot("heatmap-chart", [{
    z: d.matrix, x: d.columns, y: d.columns, type: "heatmap",
    colorscale: [[0, "#312e81"], [0.5, "#0b0e14"], [1, "#6366f1"]],
    zmin: -1, zmax: 1, text: d.matrix.map((row) => row.map((v) => v.toFixed(2))),
    texttemplate: "%{text}", textfont: { size: 9, color: "#f1f5f9" },
  }], { ...PLOTLY_LAYOUT, title: { text: "Correlation Heatmap", font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 }, yaxis: { ...PLOTLY_LAYOUT.yaxis, autorange: "reversed" },
    margin: { t: 50, r: 20, b: 120, l: 120 },
  }, PLOTLY_CONFIG);
  document.getElementById("heatmap-interp").innerHTML = d.interpretation;
}

async function computeCorrelation() {
  const v1 = document.getElementById("corr-var1").value;
  const v2 = document.getElementById("corr-var2").value;
  const d = await api("/api/correlation/pair", { method: "POST", body: { var1: v1, var2: v2 } });
  document.getElementById("corr-result").innerHTML = `
    <div class="stat-grid">${statBox("Pearson r", d.r)}${statBox("P-value", d.p_value)}</div>
    <div class="interpretation-box">${d.interpretation}</div>`;
}

async function runRegression() {
  const dep = document.getElementById("reg-dep").value;
  const indep = document.getElementById("reg-indep").value;
  const d = await api("/api/regression", { method: "POST", body: { dependent: dep, independent: indep } });
  if (d.error) { alert(d.error); return; }
  const traces = [];
  traces.push({
    x: d.x_values, y: d.actual, mode: "markers", type: "scatter", name: "Actual",
    marker: { color: "#6366f1", size: 5, opacity: 0.6 },
  });
  
  const pts = d.x_values.map((x, i) => ({ x, y: d.predicted[i] })).sort((a, b) => a.x - b.x);
  traces.push({
    x: pts.map(p => p.x),
    y: pts.map(p => p.y),
    mode: "lines",
    type: "scatter",
    name: "Predicted",
    line: { color: "#ef4444", width: 2 },
  });

  Plotly.newPlot("reg-chart", traces, { ...PLOTLY_LAYOUT,
    title: { text: `Regression: ${dep} ~ ${indep}`, font: { size: 16, color: "#f1f5f9" } },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: indep }, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: dep },
    legend: { font: { color: "#94a3b8" } },
  }, PLOTLY_CONFIG);
  
  document.getElementById("reg-results").innerHTML = `
    <div class="stat-grid mt">${statBox("R²", d.r_squared)}${statBox("Adj R²", d.adj_r_squared)}${statBox("Model P-value", d.f_pvalue)}</div>
    <div class="interpretation-box mt">${d.interpretation}</div>`;
}

async function runPrediction() {
  const dep = document.getElementById("reg-dep").value;
  const indep = document.getElementById("reg-indep").value;
  const value = parseFloat(document.getElementById("pred-value").value);
  if (isNaN(value)) { alert("Please enter a numeric value"); return; }
  const d = await api("/api/predict", { method: "POST", body: { dependent: dep, independent: indep, value } });
  document.getElementById("pred-results").innerHTML = `
    <div class="stat-grid mt"><div class="stat-item"><div class="stat-label">Predicted ${dep}</div><div class="stat-value">${d.prediction}</div></div></div>
    <div class="interpretation-box mt">${d.interpretation}</div>`;
}


