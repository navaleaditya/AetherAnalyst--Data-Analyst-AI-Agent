// State Management
let currentSessionId = null;
let pollingInterval = null;
const sessionHistoryKey = "aether_analyst_sessions";

// Store active Chart.js instances to destroy them before re-rendering
let activeCharts = {};

// DOM Elements
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const sessionList = document.getElementById("session-list");
const sessionTitle = document.getElementById("session-title");
const sessionSubtitle = document.getElementById("session-subtitle");
const downloadReportBtn = document.getElementById("download-report-btn");
const stepperSection = document.getElementById("stepper-section");
const consoleLogs = document.getElementById("console-logs");
const pipelineBadge = document.getElementById("pipeline-badge");
const welcomeScreen = document.getElementById("welcome-screen");
const dashboardGrid = document.getElementById("dashboard-grid");

// Human-In-The-Loop Approval elements
const approvalSection = document.getElementById("approval-section");
const approvalTaskType = document.getElementById("approval-task-type");
const approvalTargetCol = document.getElementById("approval-target-col");
const approvalTargetColContainer = document.getElementById("approval-target-col-container");
const approvalSubmitBtn = document.getElementById("approval-submit-btn");

// Stepper nodes
const stepNodes = {
    dataset_profiler: document.getElementById("step-profiler"),
    correlation_analysis: document.getElementById("step-correlation"),
    task_detection: document.getElementById("step-detector"),
    model_execution: document.getElementById("step-executor"),
    chart_generation: document.getElementById("step-charts"),
    saving_memory: document.getElementById("step-memory")
};

// Step order mapping for visual tracking
const stepOrder = [
    "dataset_profiler",
    "correlation_analysis",
    "task_detection",
    "model_execution",
    "chart_generation",
    "saving_memory"
];

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    loadSessionHistory();
    setupUploadHandlers();
    setupApprovalHandlers();
    
    downloadReportBtn.addEventListener("click", () => {
        if (currentSessionId) {
            window.open(`/api/report/${currentSessionId}`, '_blank');
        }
    });
});

// Drag & Drop Upload Handlers
function setupUploadHandlers() {
    uploadZone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
    
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "var(--secondary)";
    });
    
    uploadZone.addEventListener("dragleave", () => {
        uploadZone.style.borderColor = "rgba(124, 58, 237, 0.3)";
    });
    
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = "rgba(124, 58, 237, 0.3)";
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

// Approval Form Event Listeners
function setupApprovalHandlers() {
    // Show/hide target variable select based on task type
    approvalTaskType.addEventListener("change", () => {
        if (approvalTaskType.value === "classification") {
            approvalTargetColContainer.style.display = "block";
        } else {
            approvalTargetColContainer.style.display = "none";
        }
    });
    
    approvalSubmitBtn.addEventListener("click", async () => {
        if (!currentSessionId) return;
        
        const taskType = approvalTaskType.value;
        const targetColumn = approvalTargetCol.value;
        
        appendLog(`> Submitting validation choices: task_type=${taskType}, target_column=${taskType === 'classification' ? targetColumn : 'None'}.`, "system");
        
        try {
            approvalSubmitBtn.disabled = true;
            approvalSubmitBtn.innerText = "Submitting...";
            
            const response = await fetch(`/api/approve/${currentSessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    task_type: taskType,
                    target_column: taskType === "classification" ? targetColumn : null
                })
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Approval failed");
            }
            
            appendLog(`> Approved choices submitted successfully. Resuming background execution loop...`, "success");
            
            // Hide approval panel
            approvalSection.style.display = "none";
            
            // Re-start polling the background tasks
            startStatusPolling(currentSessionId, sessionTitle.innerText.replace("Analysis: ", ""));
            
        } catch (e) {
            appendLog(`> Approval error: ${e.message}`, "error");
            approvalSubmitBtn.disabled = false;
            approvalSubmitBtn.innerText = "Approve & Resume Pipeline";
        }
    });
}

// Upload Dataset
async function handleFileUpload(file) {
    const name = file.name;
    if (! (name.endsWith(".csv") || name.endsWith(".xlsx") || name.endsWith(".xls"))) {
        alert("Unsupported file type. Please upload a CSV or Excel dataset.");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", file);
    
    // Stop any active polling
    if (pollingInterval) clearInterval(pollingInterval);
    
    // UI state updates: Reset dashboard and stepper
    currentSessionId = null;
    downloadReportBtn.disabled = true;
    dashboardGrid.style.display = "none";
    welcomeScreen.style.display = "none";
    approvalSection.style.display = "none";
    stepperSection.style.display = "flex";
    
    sessionTitle.innerText = "Analyzing Dataset...";
    sessionSubtitle.innerText = `Ingesting ${name} into the agentic processing loop.`;
    pipelineBadge.innerText = "Ingesting...";
    
    // Clear logs and stepper classes
    consoleLogs.innerHTML = `<div class="log-entry log-system">> Uploading ${name} to backend...</div>`;
    resetStepperUI();
    
    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Upload failed");
        }
        
        const data = await response.json();
        currentSessionId = data.session_id;
        
        appendLog(`> File successfully uploaded. Created Session ID: ${currentSessionId}`, "success");
        appendLog(`> Ingested dataset and spawned background Celery task.`, "system");
        
        // Start polling status
        startStatusPolling(currentSessionId, name);
        
        // Save to local storage history list
        saveSessionToHistory(currentSessionId, name);
        
    } catch (e) {
        appendLog(`> Upload error: ${e.message}`, "error");
        sessionTitle.innerText = "Analysis Failed";
        sessionSubtitle.innerText = "An error occurred during dataset ingestion.";
    }
}

// Reset Stepper CSS states
function resetStepperUI() {
    Object.values(stepNodes).forEach(node => {
        node.classList.remove("active", "completed", "error");
    });
    const lines = document.querySelectorAll(".step-line");
    lines.forEach(line => line.classList.remove("active", "completed"));
}

// Show Approval UI
function showApprovalUI(statusData, sessionId) {
    pipelineBadge.innerText = "Awaiting Approval";
    pipelineBadge.style.backgroundColor = "var(--primary)";
    
    // Highlight step 3 (task detector) as active/complete
    updateStepperProgress("task_detection");
    stepNodes["task_detection"].classList.remove("active");
    stepNodes["task_detection"].classList.add("completed");
    
    // Configure inputs
    approvalTaskType.value = statusData.task_type || "classification";
    
    // Trigger task type visibility container logic
    if (approvalTaskType.value === "classification") {
        approvalTargetColContainer.style.display = "block";
    } else {
        approvalTargetColContainer.style.display = "none";
    }
    
    // Populate column options
    approvalTargetCol.innerHTML = "";
    if (statusData.df_columns && statusData.df_columns.length > 0) {
        statusData.df_columns.forEach(col => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.innerText = col;
            if (col === statusData.target_column) {
                opt.selected = true;
            }
            approvalTargetCol.appendChild(opt);
        });
    }
    
    approvalSubmitBtn.disabled = false;
    approvalSubmitBtn.innerText = "Approve & Resume Pipeline";
    
    // Reveal panel
    approvalSection.style.display = "flex";
    approvalSection.scrollIntoView({ behavior: 'smooth' });
}

// Start Status Polling
function startStatusPolling(sessionId, datasetName) {
    let lastLogLength = 0;
    
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${sessionId}`);
            if (!response.ok) throw new Error("Status query failed");
            
            const data = await response.json();
            
            // 1. Update logs console
            if (data.logs && data.logs.length > lastLogLength) {
                for (let i = lastLogLength; i < data.logs.length; i++) {
                    let type = "info";
                    if (data.logs[i].toLowerCase().includes("error") || data.logs[i].toLowerCase().includes("failed")) type = "error";
                    else if (data.logs[i].toLowerCase().includes("successfully") || data.logs[i].toLowerCase().includes("completed")) type = "success";
                    else if (data.logs[i].includes("node") || data.logs[i].includes("node...")) type = "system";
                    
                    appendLog(data.logs[i], type);
                }
                lastLogLength = data.logs.length;
            }
            
            // 2. Update pipeline badge and stepper UI
            const status = data.status;
            pipelineBadge.innerText = status.replace(/_/g, " ");
            
            if (status === "completed") {
                clearInterval(pollingInterval);
                markAllStepsCompleted();
                appendLog(`> Pipeline execution successfully completed. Loading dashboard...`, "success");
                loadResults(sessionId);
            } else if (status === "awaiting_approval") {
                clearInterval(pollingInterval);
                appendLog(`> Pipeline paused. Awaiting human-in-the-loop task validation approval.`, "system");
                showApprovalUI(data, sessionId);
            } else if (status === "error") {
                clearInterval(pollingInterval);
                markStepError(sessionId);
            } else {
                updateStepperProgress(status);
            }
            
        } catch (e) {
            clearInterval(pollingInterval);
            appendLog(`> Polling error: ${e.message}`, "error");
        }
    }, 1000);
}

// Update step CSS nodes
function updateStepperProgress(currentStatus) {
    const currentIndex = stepOrder.indexOf(currentStatus);
    if (currentIndex === -1) return;
    
    resetStepperUI();
    const lines = document.querySelectorAll(".step-line");
    
    for (let i = 0; i < stepOrder.length; i++) {
        const key = stepOrder[i];
        const node = stepNodes[key];
        
        if (i < currentIndex) {
            node.classList.add("completed");
            if (i > 0 && lines[i - 1]) lines[i - 1].classList.add("completed");
        } else if (i === currentIndex) {
            node.classList.add("active");
            if (i > 0 && lines[i - 1]) lines[i - 1].classList.add("active");
        }
    }
}

function markAllStepsCompleted() {
    Object.values(stepNodes).forEach(node => {
        node.classList.remove("active");
        node.classList.add("completed");
    });
    const lines = document.querySelectorAll(".step-line");
    lines.forEach(line => {
        line.classList.remove("active");
        line.classList.add("completed");
    });
}

function markStepError(sessionId) {
    Object.keys(stepNodes).forEach(key => {
        const node = stepNodes[key];
        if (node.classList.contains("active")) {
            node.classList.remove("active");
            node.classList.add("error");
        }
    });
    sessionTitle.innerText = "Pipeline Interrupted";
    sessionSubtitle.innerText = "The analysis agent encountered a critical error during execution.";
    pipelineBadge.innerText = "Error";
    appendLog(`> Agent execution stopped due to a fatal error.`, "error");
}

function appendLog(text, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry log-${type}`;
    entry.innerText = text;
    consoleLogs.appendChild(entry);
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// Destroy existing chart instance to prevent layout duplicates on rebuilds
function destroyChartInstance(chartId) {
    if (activeCharts[chartId]) {
        activeCharts[chartId].destroy();
        delete activeCharts[chartId];
    }
}

// Load final results from the backend and render interactive charts
async function loadResults(sessionId) {
    try {
        const response = await fetch(`/api/results/${sessionId}`);
        if (!response.ok) throw new Error("Failed to load results");
        
        const data = await response.json();
        
        // Update header title
        const filename = data.file_path.split(/[\\/]/).pop().replace(/^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}_/, "");
        sessionTitle.innerText = `Analysis: ${filename}`;
        sessionSubtitle.innerText = `Session ID: ${sessionId} | Processed ${data.df_shape[0]} rows across ${data.df_shape[1]} features`;
        
        // 1. Data Summary Profile
        document.getElementById("val-rows").innerText = data.df_shape[0].toLocaleString();
        document.getElementById("val-cols").innerText = data.df_shape[1].toLocaleString();
        document.getElementById("val-task").innerText = (data.task_type || "N/A").toUpperCase();
        document.getElementById("val-target").innerText = data.target_column || "N/A (Unsupervised)";
        
        // Populate profile table
        const profileTableBody = document.querySelector("#profile-table-summary tbody");
        profileTableBody.innerHTML = "";
        
        const profile = data.df_profile;
        if (profile && profile.data_types) {
            Object.keys(profile.data_types).forEach(colName => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${escapeHtml(colName)}</strong></td>
                    <td><span class="pipeline-badge" style="background: rgba(255,255,255,0.05); border: 1px solid var(--border-normal); text-transform: lowercase;">${profile.data_types[colName]}</span></td>
                    <td>${profile.unique_values ? profile.unique_values[colName].toLocaleString() : '-'}</td>
                    <td>${profile.missing_values && profile.missing_values[colName] > 0 ? `<span style="color: var(--error)">${profile.missing_values[colName].toLocaleString()}</span>` : '0'}</td>
                `;
                profileTableBody.appendChild(tr);
            });
        }
        
        // 2. Pearson Correlation Heatmap - Render an Interactive HTML Grid Heatmap
        const correlationChartContainer = document.getElementById("correlation-chart-container");
        correlationChartContainer.innerHTML = "";
        
        if (data.correlation_matrix && data.correlation_matrix.columns) {
            const matrix = data.correlation_matrix;
            const cols = matrix.columns;
            const vals = matrix.values;
            
            const table = document.createElement("div");
            table.className = "interactive-heatmap";
            table.style.display = "grid";
            table.style.gridTemplateColumns = `repeat(${cols.length + 1}, minmax(60px, 1fr))`;
            table.style.gap = "4px";
            table.style.margin = "12px 0";
            table.style.overflowX = "auto";
            
            // Corner cell
            const corner = document.createElement("div");
            corner.style.padding = "6px";
            corner.style.fontSize = "0.75rem";
            corner.style.color = "#A6ADC8";
            table.appendChild(corner);
            
            // Top Headers
            cols.forEach(col => {
                const header = document.createElement("div");
                header.style.padding = "6px";
                header.style.fontSize = "0.75rem";
                header.style.fontWeight = "600";
                header.style.color = "#CDD6F4";
                header.style.textOverflow = "ellipsis";
                header.style.overflow = "hidden";
                header.style.whiteSpace = "nowrap";
                header.style.textAlign = "center";
                header.innerText = col;
                table.appendChild(header);
            });
            
            // Rows
            for (let i = 0; i < cols.length; i++) {
                const rowHeader = document.createElement("div");
                rowHeader.style.padding = "6px";
                rowHeader.style.fontSize = "0.75rem";
                rowHeader.style.fontWeight = "600";
                rowHeader.style.color = "#CDD6F4";
                rowHeader.style.textOverflow = "ellipsis";
                rowHeader.style.overflow = "hidden";
                rowHeader.style.whiteSpace = "nowrap";
                rowHeader.innerText = cols[i];
                table.appendChild(rowHeader);
                
                for (let j = 0; j < cols.length; j++) {
                    const cellVal = vals[i][j];
                    const cell = document.createElement("div");
                    cell.style.padding = "10px 4px";
                    cell.style.fontSize = "0.8rem";
                    cell.style.textAlign = "center";
                    cell.style.borderRadius = "4px";
                    cell.style.fontWeight = "500";
                    cell.style.cursor = "pointer";
                    cell.innerText = cellVal.toFixed(2);
                    
                    // Style by correlation value (blue positive, red negative)
                    const absVal = Math.abs(cellVal);
                    if (cellVal >= 0) {
                        cell.style.background = `rgba(137, 180, 250, ${absVal})`; // #89B4FA
                        cell.style.color = absVal > 0.5 ? "#11111b" : "#CDD6F4";
                    } else {
                        cell.style.background = `rgba(243, 139, 168, ${absVal})`; // #F38BA8
                        cell.style.color = absVal > 0.5 ? "#11111b" : "#CDD6F4";
                    }
                    
                    // Simple interactive tooltip
                    cell.title = `Correlation [${cols[i]} × ${cols[j]}]: ${cellVal.toFixed(4)}`;
                    
                    table.appendChild(cell);
                }
            }
            correlationChartContainer.appendChild(table);
        } else {
            correlationChartContainer.innerHTML = `<div class="chart-loading">No interactive heatmap data available</div>`;
        }
        
        // Populate correlation summary bullet items
        const correlationList = document.getElementById("correlation-list");
        correlationList.innerHTML = "";
        if (data.correlation_summary && data.correlation_summary.length > 0) {
            data.correlation_summary.forEach(item => {
                const li = document.createElement("li");
                li.innerHTML = `Feature pair <strong>${escapeHtml(item.feature1)}</strong> and <strong>${escapeHtml(item.feature2)}</strong> (coeff: <strong>${item.correlation.toFixed(3)}</strong>)`;
                correlationList.appendChild(li);
            });
        } else {
            correlationList.innerHTML = "<li>No significant numeric correlations found.</li>";
        }
        
        // 3. ML Task Detector callout
        const badge = document.getElementById("reasoning-badge");
        badge.innerText = data.task_type;
        if (data.task_type === "classification") {
            badge.style.backgroundColor = "var(--primary)";
        } else if (data.task_type === "clustering") {
            badge.style.backgroundColor = "var(--secondary)";
        } else {
            badge.style.backgroundColor = "var(--accent)";
        }
        
        document.getElementById("reasoning-text").innerText = data.reasoning || "Reasoning analysis missing.";
        
        // 4. Populate model metrics
        const metricsGrid = document.getElementById("metrics-grid");
        metricsGrid.innerHTML = "";
        
        const results = data.ml_results;
        if (results && results.metrics) {
            const m = results.metrics;
            if (data.task_type === "classification") {
                addMetricCard(metricsGrid, "Accuracy Score", `${(m.accuracy * 100).toFixed(2)}%`);
                addMetricCard(metricsGrid, "Precision (Weighted)", `${(m.precision * 100).toFixed(2)}%`);
                addMetricCard(metricsGrid, "Recall (Weighted)", `${(m.recall * 100).toFixed(2)}%`);
                addMetricCard(metricsGrid, "F1-Score (Weighted)", `${(m.f1_score * 100).toFixed(2)}%`);
            } else if (data.task_type === "clustering") {
                addMetricCard(metricsGrid, "Best Clusters (k)", m.best_k);
                addMetricCard(metricsGrid, "Silhouette Score", m.silhouette_score.toFixed(4));
                
                if (results.cluster_sizes) {
                    Object.keys(results.cluster_sizes).forEach(cId => {
                        addMetricCard(metricsGrid, `Cluster ${cId} Size`, `${results.cluster_sizes[cId]} records`);
                    });
                }
            } else if (data.task_type === "anomaly") {
                addMetricCard(metricsGrid, "Anomaly Rate", `${(m.anomaly_rate * 100).toFixed(2)}%`);
                addMetricCard(metricsGrid, "Anomalies Flagged", `${m.anomalies_count} records`);
                addMetricCard(metricsGrid, "Normal Records", `${m.total_records - m.anomalies_count} records`);
                addMetricCard(metricsGrid, "Mean Decision Score", m.mean_anomaly_score.toFixed(4));
            }
        }
        
        // 5. Model visual outputs (Chart.js Interactive Charts)
        const modelChartTitle = document.getElementById("model-chart-title");
        const modelChartContainer = document.getElementById("model-chart-container");
        modelChartContainer.innerHTML = "";
        
        const cmBox = document.getElementById("confusion-matrix-box");
        const cmContainer = document.getElementById("confusion-matrix-container");
        cmContainer.innerHTML = "";
        cmBox.style.display = "none";
        
        // Destroy any active charts
        destroyChartInstance("model_insights");
        destroyChartInstance("confusion_matrix");
        
        if (data.task_type === "classification") {
            modelChartTitle.innerText = "Model Feature Attributions (SHAP/Native)";
            
            if (results && results.feature_importances) {
                const importances = results.feature_importances;
                const features = Object.keys(importances).slice(0, 10);
                const values = Object.values(importances).slice(0, 10);
                
                const canvas = document.createElement("canvas");
                canvas.id = "chart-feature-importance";
                modelChartContainer.appendChild(canvas);
                
                const ctx = canvas.getContext("2d");
                activeCharts["model_insights"] = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: features,
                        datasets: [{
                            label: 'Relative Attribution',
                            data: values,
                            backgroundColor: values.map((_, i) => i === 0 ? '#CBA6F7' : '#89B4FA'), // Highlight top
                            borderRadius: 4
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: '#252538',
                                titleColor: '#CDD6F4',
                                bodyColor: '#CDD6F4',
                                borderColor: '#44445c',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                grid: { color: '#313244' },
                                ticks: { color: '#CDD6F4' }
                            },
                            y: {
                                grid: { display: false },
                                ticks: { color: '#CDD6F4', font: { weight: 'bold' } }
                            }
                        }
                    }
                });
            } else {
                modelChartContainer.innerHTML = `<div class="chart-loading">No feature importances found</div>`;
            }
            
            // Confusion Matrix
            if (results && results.confusion_matrix) {
                cmBox.style.display = "flex";
                
                const mapping = results.target_mapping || {};
                const labels = [mapping[0] || "Class 0", mapping[1] || "Class 1"];
                if (results.confusion_matrix.length > 2) {
                    labels.length = 0;
                    for (let i = 0; i < results.confusion_matrix.length; i++) {
                        labels.push(mapping[i] || `Class ${i}`);
                    }
                }
                
                const table = document.createElement("div");
                table.style.display = "grid";
                table.style.gridTemplateColumns = `repeat(${labels.length + 1}, 1fr)`;
                table.style.gap = "6px";
                table.style.margin = "12px 0";
                
                // Corner
                const corner = document.createElement("div");
                corner.style.padding = "8px";
                corner.style.fontSize = "0.8rem";
                corner.style.color = "#A6ADC8";
                corner.innerHTML = "True \\ Pred";
                table.appendChild(corner);
                
                // Headers
                labels.forEach(l => {
                    const cell = document.createElement("div");
                    cell.style.padding = "8px";
                    cell.style.fontSize = "0.85rem";
                    cell.style.fontWeight = "bold";
                    cell.style.textAlign = "center";
                    cell.style.color = "#CDD6F4";
                    cell.innerText = l;
                    table.appendChild(cell);
                });
                
                // Grid Content
                const cm = results.confusion_matrix;
                const maxVal = Math.max(...cm.flat());
                for (let i = 0; i < labels.length; i++) {
                    const rowHeader = document.createElement("div");
                    rowHeader.style.padding = "8px";
                    rowHeader.style.fontSize = "0.85rem";
                    rowHeader.style.fontWeight = "bold";
                    rowHeader.style.color = "#CDD6F4";
                    rowHeader.innerText = labels[i];
                    table.appendChild(rowHeader);
                    
                    for (let j = 0; j < labels.length; j++) {
                        const cellVal = cm[i][j];
                        const cell = document.createElement("div");
                        cell.style.padding = "14px 4px";
                        cell.style.textAlign = "center";
                        cell.style.borderRadius = "6px";
                        cell.style.fontWeight = "600";
                        cell.innerText = cellVal;
                        
                        // Diagonal hits are correct predictions -> highlight green/blue, incorrect -> red
                        if (i === j) {
                            cell.style.background = `rgba(166, 227, 161, ${cellVal / (maxVal || 1)})`; // #A6E3A1
                            cell.style.color = (cellVal / (maxVal || 1)) > 0.5 ? "#11111b" : "#CDD6F4";
                        } else {
                            cell.style.background = `rgba(243, 139, 168, ${cellVal / (maxVal || 1)})`; // #F38BA8
                            cell.style.color = (cellVal / (maxVal || 1)) > 0.5 ? "#11111b" : "#CDD6F4";
                        }
                        
                        table.appendChild(cell);
                    }
                }
                cmContainer.appendChild(table);
            }
            
        } else if (data.task_type === "clustering") {
            modelChartTitle.innerText = "PCA Cluster Scatter Plot Projection";
            
            if (results && results.pca_projection) {
                const proj = results.pca_projection;
                const preds = results.predictions;
                
                // Group points by cluster label
                const datasetsMap = {};
                for (let i = 0; i < proj.length; i++) {
                    const label = preds[i];
                    if (!datasetsMap[label]) {
                        datasetsMap[label] = {
                            label: `Cluster ${label}`,
                            data: [],
                            backgroundColor: getClusterColor(label),
                            pointRadius: 5,
                            hoverRadius: 7
                        };
                    }
                    datasetsMap[label].data.push({ x: proj[i][0], y: proj[i][1] });
                }
                
                const canvas = document.createElement("canvas");
                canvas.id = "chart-clustering-scatter";
                modelChartContainer.appendChild(canvas);
                
                const ctx = canvas.getContext("2d");
                activeCharts["model_insights"] = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: Object.values(datasetsMap)
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            tooltip: {
                                backgroundColor: '#252538',
                                titleColor: '#CDD6F4',
                                bodyColor: '#CDD6F4',
                                borderColor: '#44445c',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Principal Component 1', color: '#CDD6F4' },
                                grid: { color: '#313244' },
                                ticks: { color: '#CDD6F4' }
                            },
                            y: {
                                title: { display: true, text: 'Principal Component 2', color: '#CDD6F4' },
                                grid: { color: '#313244' },
                                ticks: { color: '#CDD6F4' }
                            }
                        }
                    }
                });
            } else {
                modelChartContainer.innerHTML = `<div class="chart-loading">No PCA projections found</div>`;
            }
            
        } else if (data.task_type === "anomaly") {
            modelChartTitle.innerText = "PCA Outlier Overlay Scatter Plot Projection";
            
            if (results && results.pca_projection) {
                const proj = results.pca_projection;
                const preds = results.predictions;
                
                const normalData = [];
                const anomalyData = [];
                
                for (let i = 0; i < proj.length; i++) {
                    if (preds[i] === 1) {
                        normalData.push({ x: proj[i][0], y: proj[i][1] });
                    } else {
                        anomalyData.push({ x: proj[i][0], y: proj[i][1] });
                    }
                }
                
                const canvas = document.createElement("canvas");
                canvas.id = "chart-anomaly-scatter";
                modelChartContainer.appendChild(canvas);
                
                const ctx = canvas.getContext("2d");
                activeCharts["model_insights"] = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            {
                                label: 'Normal',
                                data: normalData,
                                backgroundColor: '#89B4FA',
                                pointRadius: 4,
                                hoverRadius: 6
                            },
                            {
                                label: 'Anomaly',
                                data: anomalyData,
                                backgroundColor: '#F38BA8',
                                pointRadius: 7,
                                pointStyle: 'rectRot',
                                hoverRadius: 9
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            tooltip: {
                                backgroundColor: '#252538',
                                titleColor: '#CDD6F4',
                                bodyColor: '#CDD6F4',
                                borderColor: '#44445c',
                                borderWidth: 1
                            }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Principal Component 1', color: '#CDD6F4' },
                                grid: { color: '#313244' },
                                ticks: { color: '#CDD6F4' }
                            },
                            y: {
                                title: { display: true, text: 'Principal Component 2', color: '#CDD6F4' },
                                grid: { color: '#313244' },
                                ticks: { color: '#CDD6F4' }
                            }
                        }
                    }
                });
            } else {
                modelChartContainer.innerHTML = `<div class="chart-loading">No PCA projections found</div>`;
            }
        }
        
        // Enable report PDF download button
        downloadReportBtn.disabled = false;
        
        // Show grid dashboard
        dashboardGrid.style.display = "grid";
        
    } catch (e) {
        alert("Failed to load final dashboard data: " + e.message);
    }
}

// Map cluster IDs to beautiful pastel colors
function getClusterColor(id) {
    const colors = ['#CBA6F7', '#A6E3A1', '#FAB387', '#89B4FA', '#F9E2AF'];
    return colors[id % colors.length];
}

function addMetricCard(container, label, value) {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
        <span class="metric-card-label">${label}</span>
        <span class="metric-card-value">${value}</span>
    `;
    container.appendChild(card);
}

// Session History Local Storage Management
function loadSessionHistory() {
    const list = getHistoryList();
    sessionList.innerHTML = "";
    
    if (list.length === 0) {
        sessionList.innerHTML = `<li class="empty-history">No analysis sessions yet</li>`;
        return;
    }
    
    list.forEach(item => {
        const li = document.createElement("li");
        li.className = "session-item";
        if (currentSessionId === item.id) li.classList.add("active");
        
        li.innerHTML = `
            <span class="session-item-name">${escapeHtml(item.name)}</span>
            <span class="session-item-id">${item.id.substring(0, 8)}...</span>
        `;
        
        li.addEventListener("click", () => {
            if (pollingInterval) clearInterval(pollingInterval);
            
            // Switch session
            currentSessionId = item.id;
            
            // Set active class
            document.querySelectorAll(".session-item").forEach(node => node.classList.remove("active"));
            li.classList.add("active");
            
            // Hide elements
            stepperSection.style.display = "none";
            welcomeScreen.style.display = "none";
            approvalSection.style.display = "none";
            
            loadResults(item.id);
        });
        
        sessionList.appendChild(li);
    });
}

function getHistoryList() {
    try {
        const data = localStorage.getItem(sessionHistoryKey);
        return data ? JSON.parse(data) : [];
    } catch (e) {
        return [];
    }
}

function saveSessionToHistory(id, name) {
    const list = getHistoryList();
    const index = list.findIndex(item => item.id === id);
    if (index === -1) {
        list.unshift({ id, name, timestamp: Date.now() });
    }
    if (list.length > 10) list.pop();
    localStorage.setItem(sessionHistoryKey, JSON.stringify(list));
    loadSessionHistory();
}

// Security: Escape HTML inputs
function escapeHtml(text) {
    if (!text) return "";
    return text.toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
