let currentData = null;
let currentProfile = null;
let currentCharts = null;

// File upload handling
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");

uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    handleFileUpload();
  }
});

fileInput.addEventListener("change", handleFileUpload);

async function handleFileUpload() {
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  showLoading(true);
  showNotification("Uploading and analysing your data...", "info");

  try {
    const response = await fetch("/upload", { method: "POST", body: formData });
    const result = await response.json();

    if (result.success) {
      currentData = result;
      currentProfile = result.profile;
      currentCharts = result.charts;
      displayResults(result);
      showNotification("Data uploaded and analysed successfully!", "success");
    } else {
      showNotification("Error: " + result.error, "danger");
    }
  } catch (error) {
    showNotification("Error uploading file: " + error.message, "danger");
  } finally {
    showLoading(false);
  }
}

function setupSheetSelector(sheetInfo) {
  const container = document.getElementById("sheetSelectorContainer");
  const select = document.getElementById("sheetSelector");

  if (!sheetInfo || sheetInfo.sheet_count <= 1) {
    container.style.display = "none";
    return;
  }

  container.style.display = "block";
  document.getElementById("activeSheetName").textContent =
    sheetInfo.active_sheet;
  document.getElementById("sheetCount").textContent = sheetInfo.sheet_count;

  select.innerHTML = "";
  sheetInfo.sheet_names.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === sheetInfo.active_sheet) opt.selected = true;
    select.appendChild(opt);
  });

  select.onchange = async (e) => {
    const newSheet = e.target.value;
    if (newSheet === sheetInfo.active_sheet) return;
    await switchSheet(newSheet);
  };
}

async function switchSheet(sheetName) {
  showLoading(true);
  showNotification(`Switching to sheet: ${sheetName}...`, "info");

  try {
    const resp = await fetch("/switch_sheet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sheet_name: sheetName }),
    });
    const data = await resp.json();

    if (!data.success) {
      showNotification("Error: " + data.error, "danger");
      return;
    }

    currentData = data;
    currentProfile = data.profile;
    currentCharts = data.charts;

    document.getElementById("rowCount").textContent = data.rows;
    document.getElementById("colCount").textContent = data.columns;
    document.getElementById("missingCount").textContent =
      (data.profile.missing_values &&
        data.profile.missing_values.total_missing) ||
      0;
    const qs =
      data.profile.data_quality && data.profile.data_quality.quality_score;
    document.getElementById("qualityScore").textContent =
      qs === null || qs === undefined ? "—" : qs + "%";

    document.getElementById("preview-table").innerHTML = data.preview;
    displayCharts(data.charts);
    displayProfile(data.profile);
    setupSheetSelector(data.sheet_info);

    showNotification(`Loaded sheet: ${sheetName}`, "success");
  } catch (err) {
    showNotification("Error switching sheet: " + err.message, "danger");
  } finally {
    showLoading(false);
  }
}

function displayResults(data) {
  document.getElementById("resultsSection").style.display = "block";

  document.getElementById("rowCount").textContent = data.rows;
  document.getElementById("colCount").textContent = data.columns;
  document.getElementById("missingCount").textContent =
    data.profile.missing_values.total_missing || 0;
  document.getElementById("qualityScore").textContent =
    data.profile.data_quality.quality_score + "%";

  document.getElementById("preview-table").innerHTML = data.preview;

  try {
    displayCharts(data.charts);
  } catch (e) {
    console.error("displayCharts failed:", e);
  }
  try {
    displayProfile(data.profile);
  } catch (e) {
    console.error("displayProfile failed:", e);
  }

  setupSheetSelector(data.sheet_info);

  setTimeout(runAutoML, 1000);
}

function displayCharts(charts) {
  const container = document.getElementById("chartsContainer");
  container.innerHTML = "";
  if (!charts || Object.keys(charts).length === 0) {
    container.innerHTML =
      '<div class="alert alert-info">No charts generated automatically.</div>';
    return;
  }

  Object.keys(charts).forEach((key, index) => {
    const chart = charts[key];
    const col = document.createElement("div");
    col.className = "col-md-6";

    const chartDivId = `chart-${key}-${index}`;
    col.innerHTML = `
      <div class="chart-container">
        <h6>${chart.config.title || "Chart " + (index + 1)}</h6>
        <div id="${chartDivId}" style="width:100%;height:450px;"></div>
      </div>`;
    container.appendChild(col);

    try {
      const figure =
        typeof chart.figure === "string"
          ? JSON.parse(chart.figure)
          : chart.figure;
      Plotly.newPlot(chartDivId, figure.data, figure.layout, {
        responsive: true,
      });
    } catch (err) {
      console.error(`Failed to render chart ${key}:`, err);
      document.getElementById(chartDivId).innerHTML =
        `<div class="alert alert-danger">Render error: ${err.message}</div>`;
    }
  });
}

function displayProfile(profile) {
  const container = document.getElementById("profileContent");
  let html = '<div class="row">';

  html += `
    <div class="col-md-6">
      <div class="card mb-3">
        <div class="card-header"><strong>Dataset Overview</strong></div>
        <div class="card-body">
          <p><strong>Rows:</strong> ${profile.basic_info.rows}</p>
          <p><strong>Columns:</strong> ${profile.basic_info.columns}</p>
          <p><strong>Memory Usage:</strong> ${profile.basic_info.memory_usage.toFixed(2)} MB</p>
          <p><strong>Data Types:</strong> ${Object.entries(
            profile.basic_info.data_types,
          )
            .map(([col, type]) => `${col}: ${type}`)
            .join("<br>")}</p>
        </div>
      </div>
    </div>`;

  html += `
    <div class="col-md-6">
      <div class="card mb-3">
        <div class="card-header"><strong>Missing Values</strong></div>
        <div class="card-body">
          <p><strong>Total Missing:</strong> ${profile.missing_values.total_missing}</p>
          <p><strong>Columns with Missing:</strong> ${profile.missing_values.columns_with_missing}</p>
          ${Object.entries(profile.missing_values.missing_by_column || {})
            .map(
              ([col, count]) =>
                `<small>${col}: ${count} (${(profile.missing_values.missing_percentage_by_column[col] || 0).toFixed(1)}%)</small><br>`,
            )
            .join("")}
        </div>
      </div>
    </div>`;

  html += `
    <div class="col-12">
      <div class="card">
        <div class="card-header"><strong>Data Quality Recommendations</strong></div>
        <div class="card-body">
          ${profile.data_quality.recommendations
            .map(
              (rec) =>
                `<div class="alert alert-${rec.type === "warning" ? "warning" : rec.type === "info" ? "info" : "success"} alert-sm">
                 <i class="fas fa-${rec.type === "warning" ? "exclamation-triangle" : rec.type === "info" ? "info-circle" : "check-circle"}"></i>
                 ${rec.message}
               </div>`,
            )
            .join("")}
        </div>
      </div>
    </div>`;

  html += "</div>";
  container.innerHTML = html;
}

async function runAutoML() {
  const container = document.getElementById("automlResults");
  container.innerHTML =
    '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p>Training models...</p></div>';

  try {
    const response = await fetch("/analyse", { method: "POST" });
    const result = await response.json();

    if (result.success) {
      displayAutoMLResults(result);
      showNotification("AutoML training completed!", "success");
    } else {
      container.innerHTML = `<div class="alert alert-danger">Error: ${result.error}</div>`;
    }
  } catch (error) {
    container.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
  }
}

function displayAutoMLResults(data) {
  const container = document.getElementById("automlResults");
  const report = data.performance_report;

  let html = `
    <div class="card mb-3">
      <div class="card-header">
        <strong>Problem Type:</strong> <span class="badge badge-ai">${data.problem_type}</span>
        <strong class="ms-3">Target Column:</strong> <span class="badge bg-primary">${data.target_column}</span>
      </div>
      <div class="card-body">
        <p><strong>Training Samples:</strong> ${report.training_samples}</p>
        <p><strong>Test Samples:</strong> ${report.test_samples}</p>
        <p><strong>Features:</strong> ${report.feature_count}</p>
        <p><strong>Best Model:</strong> <span class="badge bg-success">${report.best_model}</span></p>
        <p><strong>Best Score:</strong> ${(report.best_score * 100).toFixed(2)}%</p>
      </div>
    </div>`;

  html += `<h6>Model Performance Comparison</h6><div class="row">`;
  Object.entries(report.model_performance).forEach(([name, perf]) => {
    const isBest = name === report.best_model;
    html += `
      <div class="col-md-4">
        <div class="card model-card ${isBest ? "best" : ""}">
          <div class="card-body">
            <h6>${name} ${isBest ? '<span class="badge bg-success">⭐ Best</span>' : ""}</h6>
            <p><strong>Score:</strong> ${(perf.score * 100).toFixed(2)}%</p>
            <p><strong>Metric:</strong> ${perf.metric}</p>
          </div>
        </div>
      </div>`;
  });
  html += "</div>";

  if (report.feature_importance && report.feature_importance.length > 0) {
    html += `<h6 class="mt-4">Top 10 Most Important Features</h6><div class="list-group">`;
    report.feature_importance.slice(0, 10).forEach(([feature, importance]) => {
      const percentage = (importance * 100).toFixed(1);
      html += `
        <div class="list-group-item">
          <div class="d-flex justify-content-between align-items-center">
            <span>${feature}</span>
            <span class="badge bg-primary">${percentage}%</span>
          </div>
          <div class="feature-importance-bar" style="width: ${percentage}%;"></div>
        </div>`;
    });
    html += "</div>";
  }

  container.innerHTML = html;

  // Direct call replaces the old monkey-patch below.
  showPredictTab();
}

function refreshCharts() {
  if (currentCharts) {
    displayCharts(currentCharts);
    showNotification("Charts refreshed!", "info");
  }
}

function showLoading(show) {
  document.getElementById("loadingSpinner").style.display = show
    ? "block"
    : "none";
}

function showNotification(message, type = "info") {
  const container = document.getElementById("notification");
  const colors = {
    success: "alert-success",
    danger: "alert-danger",
    warning: "alert-warning",
    info: "alert-info",
  };

  container.innerHTML = `
    <div class="alert ${colors[type] || "alert-info"} alert-dismissible fade show" role="alert">
      ${message}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>`;

  setTimeout(() => {
    const alert = container.querySelector(".alert");
    if (alert) {
      alert.classList.remove("show");
      setTimeout(() => (container.innerHTML = ""), 300);
    }
  }, 5000);
}

// Prediction tab
let modelInfo = null;

async function loadModelInfo() {
  try {
    const resp = await fetch("/model_info");
    if (!resp.ok) {
      document.getElementById("predictContent").style.display = "none";
      document.getElementById("predictIntro").style.display = "block";
      return null;
    }
    const data = await resp.json();
    if (!data.success) return null;
    modelInfo = data;
    return data;
  } catch (e) {
    console.error("loadModelInfo failed:", e);
    return null;
  }
}

// REFACTOR: single definition (the earlier duplicate was deleted).
function buildPredictForm(featureCols) {
  const container = document.getElementById("predictForm");
  container.innerHTML = "";
  featureCols.forEach((col) => {
    const colDiv = document.createElement("div");
    colDiv.className = "col-md-4 mb-3";
    colDiv.innerHTML = `
      <label class="form-label"><small>${col}</small></label>
      <input type="text" class="form-control form-control-sm"
             data-feature="${col}" placeholder="value">`;
    container.appendChild(colDiv);
  });
}

async function showPredictTab() {
  const info = await loadModelInfo();
  if (!info) {
    document.getElementById("predictIntro").style.display = "block";
    document.getElementById("predictContent").style.display = "none";
    return;
  }
  document.getElementById("predictIntro").style.display = "none";
  document.getElementById("predictContent").style.display = "block";
  document.getElementById("modelBadge").textContent =
    info.best_model_name + " (" + (info.best_score * 100).toFixed(1) + "%)";
  document.getElementById("predictTarget").textContent = info.target_column;
  buildPredictForm(info.feature_columns);
}

document
  .getElementById("clearSessionBtn")
  .addEventListener("click", async () => {
    if (!confirm("Clear the current session and start over?")) return;
    try {
      await fetch("/clear_session", { method: "POST" });
      window.location.reload();
    } catch (e) {
      console.error("Clear session failed:", e);
    }
  });

document.getElementById("fillExampleBtn").addEventListener("click", () => {
  if (!modelInfo || !modelInfo.sample_row) return;
  document.querySelectorAll("[data-feature]").forEach((inp) => {
    const col = inp.dataset.feature;
    const val = modelInfo.sample_row[col];
    inp.value = val === null || val === undefined ? "" : val;
  });
});

document.getElementById("downloadModelBtn").addEventListener("click", () => {
  window.location.href = "/download_model";
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const inputs = document.querySelectorAll("[data-feature]");
  const row = {};
  inputs.forEach((inp) => {
    const v = inp.value.trim();
    if (v === "") return;
    const num = Number(v);
    row[inp.dataset.feature] = isNaN(num) ? v : num;
  });

  const out = document.getElementById("predictResults");
  out.innerHTML =
    '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Predicting...</div>';

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: [row] }),
    });
    const data = await resp.json();
    if (!data.success) {
      out.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
      return;
    }
    const pred = data.predictions[0];
    const prob = data.probabilities ? data.probabilities[0] : null;
    let probHtml = "";
    if (prob) {
      const labels =
        data.problem_type === "classification"
          ? (modelInfo && modelInfo.target_labels) || ["Class 0", "Class 1"]
          : [];
      probHtml =
        '<ul class="list-unstyled mb-0">' +
        prob
          .map(
            (p, i) =>
              `<li>${labels[i] || "Class " + i}: <strong>${(p * 100).toFixed(1)}%</strong></li>`,
          )
          .join("") +
        "</ul>";
    }
    out.innerHTML = `
      <div class="alert alert-success">
        <h5 class="mb-1">Prediction: <strong>${pred}</strong></h5>
        <small class="text-muted">Target: ${data.target_column}</small>
        ${probHtml ? '<hr class="my-2">' + probHtml : ""}
      </div>`;
  } catch (e) {
    out.innerHTML = `<div class="alert alert-danger">${e.message}</div>`;
  }
});

document.getElementById("predict-tab").addEventListener("shown.bs.tab", () => {
  showPredictTab();
});

// Handle page unload.
window.addEventListener("beforeunload", async () => {
  try {
    await fetch("/clear_session", { method: "POST" });
  } catch (e) {
    // Ignore
  }
});
