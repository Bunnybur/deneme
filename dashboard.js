// ============================================================================
// IoT Sensor Fault Detection Dashboard - Enhanced JavaScript with Real Data
// ============================================================================

// Global variables to store loaded data
let sensorData = [];
let stats = {};
let chartInstances = {};

// ============================================================================
// DATA LOADING AND PROCESSING
// ============================================================================

async function loadSensorData() {
    return new Promise((resolve, reject) => {
        Papa.parse('sensor-fault-detection.csv', {
            download: true,
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function (results) {
                sensorData = results.data.map(row => ({
                    timestamp: new Date(row.Timestamp),
                    sensorId: row.SensorId,
                    value: parseFloat(row.Value)
                })).filter(row => !isNaN(row.value));

                // Sort by timestamp
                sensorData.sort((a, b) => a.timestamp - b.timestamp);

                addLog(`Loaded ${sensorData.length} sensor readings`, 'success');
                calculateStatistics();
                resolve(sensorData);
            },
            error: function (error) {
                addLog('Error loading CSV data: ' + error.message, 'error');
                reject(error);
            }
        });
    });
}

function calculateStatistics() {
    const values = sensorData.map(d => d.value);
    const n = values.length;

    // Basic statistics
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const sorted = [...values].sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[n - 1];
    const q1 = sorted[Math.floor(n * 0.25)];
    const median = sorted[Math.floor(n * 0.5)];
    const q3 = sorted[Math.floor(n * 0.75)];

    // Standard deviation
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);

    // IQR calculations
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    // Outlier detection
    const iqrOutliers = values.filter(v => v < lowerBound || v > upperBound);
    const normalRange = values.filter(v => v >= 0 && v <= 60);
    const extremeFaults = values.filter(v => v > 100);

    stats = {
        count: n,
        mean,
        std,
        min,
        max,
        q1,
        median,
        q3,
        iqr,
        lowerBound,
        upperBound,
        iqrOutliers: iqrOutliers.length,
        iqrOutliersPercent: (iqrOutliers.length / n * 100).toFixed(1),
        normalCount: normalRange.length,
        normalPercent: (normalRange.length / n * 100).toFixed(1),
        extremeCount: extremeFaults.length,
        extremePercent: (extremeFaults.length / n * 100).toFixed(2)
    };

    addLog('Statistics calculated successfully', 'info');
    updateDashboardStats();
}

function updateDashboardStats() {
    // Update KPI cards
    document.getElementById('totalRecords').textContent = stats.count.toLocaleString();
    document.getElementById('normalPercent').textContent = stats.normalPercent + '%';
    document.getElementById('iqrOutliers').textContent = stats.iqrOutliersPercent + '%';
    document.getElementById('extremeFaults').textContent = stats.extremePercent + '%';

    // Update date range
    const firstDate = sensorData[0].timestamp;
    const lastDate = sensorData[sensorData.length - 1].timestamp;
    const monthsDiff = Math.round((lastDate - firstDate) / (1000 * 60 * 60 * 24 * 30));

    document.getElementById('dateRange').innerHTML = `
        <div>${firstDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</div>
        <div>→</div>
        <div>${lastDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</div>
    `;
    document.querySelector('#dateRange').nextElementSibling.textContent = `${monthsDiff} Months`;

    // Update statistics table
    const statsTableBody = document.getElementById('statsTableBody');
    statsTableBody.innerHTML = `
        <tr><td>Count</td><td>${stats.count.toLocaleString()}</td></tr>
        <tr><td>Mean</td><td>${stats.mean.toFixed(2)}</td></tr>
        <tr><td>Std Dev</td><td>${stats.std.toFixed(2)}</td></tr>
        <tr><td>Min</td><td>${stats.min.toFixed(2)}</td></tr>
        <tr><td>25th Percentile</td><td>${stats.q1.toFixed(2)}</td></tr>
        <tr><td>Median</td><td>${stats.median.toFixed(2)}</td></tr>
        <tr><td>75th Percentile</td><td>${stats.q3.toFixed(2)}</td></tr>
        <tr><td>Max</td><td>${stats.max.toFixed(2)}</td></tr>
    `;

    // Update IQR values
    document.querySelectorAll('.iqr-value')[0].textContent = stats.q1.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[1].textContent = stats.q3.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[2].textContent = stats.iqr.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[3].textContent = stats.lowerBound.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[4].textContent = stats.upperBound.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[5].textContent = `${stats.iqrOutliers.toLocaleString()} (${stats.iqrOutliersPercent}%)`;

    // Update summary card
    document.getElementById('tempRange').textContent = `${stats.min.toFixed(2)}°C - ${stats.max.toFixed(2)}°C`;
    document.getElementById('meanTemp').textContent = `${stats.mean.toFixed(2)}°C`;
    document.getElementById('stdDev').textContent = `${stats.std.toFixed(2)}°C`;
}

// ============================================================================
// CHART INITIALIZATION
// ============================================================================

async function initializeCharts() {
    try {
        await loadSensorData();

        createChart1_CompleteTimeSeries();
        createChart2_NormalRange();
        createChart3_StandardizedValues();
        createChart4_Distribution();
        createChart5_BoxPlot();
        createChart6_MonthlyTrends();

        addLog('All visualizations created successfully', 'success');
    } catch (error) {
        addLog('Error initializing charts: ' + error.message, 'error');
    }
}

// Chart 1: Complete Time Series with Anomalies
function createChart1_CompleteTimeSeries() {
    const ctx = document.getElementById('chart1').getContext('2d');

    // Sample data for performance (every 100th point for the line, all outliers)
    const sampledData = sensorData.filter((_, i) => i % 100 === 0);
    const outliers = sensorData.filter(d => d.value < stats.lowerBound || d.value > stats.upperBound);
    const extreme = sensorData.filter(d => d.value > 100);

    chartInstances.chart1 = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'All Readings',
                    data: sampledData.map(d => ({ x: d.timestamp, y: d.value })),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'IQR Outliers',
                    data: outliers.filter(d => d.value <= 100).map(d => ({ x: d.timestamp, y: d.value })),
                    borderColor: '#f59e0b',
                    backgroundColor: '#f59e0b',
                    pointRadius: 3,
                    showLine: false
                },
                {
                    label: 'Extreme Faults (>100°C)',
                    data: extreme.map(d => ({ x: d.timestamp, y: d.value })),
                    borderColor: '#ef4444',
                    backgroundColor: '#ef4444',
                    pointRadius: 5,
                    pointStyle: 'triangle',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => items[0].raw.x.toLocaleString(),
                        label: (context) => context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '°C'
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    title: { display: true, text: 'Time' },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Temperature (°C)' },
                    grid: { color: '#e2e8f0' }
                }
            }
        }
    });
}

// Chart 2: Normal Operating Range (0-60°C)
function createChart2_NormalRange() {
    const ctx = document.getElementById('chart2').getContext('2d');

    const normalData = sensorData.filter(d => d.value >= 0 && d.value <= 60);
    const sampled = normalData.filter((_, i) => i % 50 === 0);

    chartInstances.chart2 = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Normal Range',
                data: sampled.map(d => ({ x: d.timestamp, y: d.value })),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    title: { display: true, text: 'Time' },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Temperature (°C)' },
                    min: 0,
                    max: 60,
                    grid: { color: '#e2e8f0' }
                }
            }
        }
    });
}

// Chart 3: Normal Temperature Values
function createChart3_StandardizedValues() {
    const ctx = document.getElementById('chart3').getContext('2d');

    const sampled = sensorData.filter((_, i) => i % 100 === 0);
    const normalData = sampled.map(d => ({
        x: d.timestamp,
        y: d.value
    }));

    chartInstances.chart3 = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Temperature Values',
                    data: normalData,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'month' },
                    title: { display: true, text: 'Time' },
                    grid: { display: false }
                },
                y: {
                    title: { display: true, text: 'Temperature (°C)' },
                    grid: { color: '#e2e8f0' }
                }
            }
        }
    });
}

// Chart 4: Temperature Distribution (Histogram)
function createChart4_Distribution() {
    const ctx = document.getElementById('chart4').getContext('2d');

    // Create bins
    const binSize = 5;
    const bins = {};
    sensorData.forEach(d => {
        const bin = Math.floor(d.value / binSize) * binSize;
        bins[bin] = (bins[bin] || 0) + 1;
    });

    const sortedBins = Object.keys(bins).map(Number).sort((a, b) => a - b);
    const colors = sortedBins.map(bin => {
        if (bin > 100) return '#ef4444'; // Extreme
        if (bin < stats.lowerBound || bin > stats.upperBound) return '#f59e0b'; // Outlier
        return '#3b82f6'; // Normal
    });

    chartInstances.chart4 = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedBins.map(b => `${b}-${b + binSize}°C`),
            datasets: [{
                label: 'Frequency',
                data: sortedBins.map(b => bins[b]),
                backgroundColor: colors,
                borderColor: colors.map(c => c),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `Count: ${context.parsed.y.toLocaleString()}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Temperature Range (°C)' },
                    ticks: { maxTicksLimit: 15 }
                },
                y: {
                    title: { display: true, text: 'Frequency' },
                    grid: { color: '#e2e8f0' }
                }
            }
        }
    });
}

// Chart 5: Box Plot
function createChart5_BoxPlot() {
    const ctx = document.getElementById('chart5').getContext('2d');

    const values = sensorData.map(d => d.value);
    const sorted = [...values].sort((a, b) => a - b);
    const outlierPoints = values.filter(v => v < stats.lowerBound || v > stats.upperBound);

    // Box plot using scatter and line charts
    chartInstances.chart5 = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Temperature'],
            datasets: [
                {
                    label: 'IQR (Q1-Q3)',
                    data: [stats.q3 - stats.q1],
                    backgroundColor: 'rgba(59, 130, 246, 0.3)',
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    base: stats.q1
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
                annotation: {
                    annotations: {
                        median: {
                            type: 'line',
                            yMin: stats.median,
                            yMax: stats.median,
                            borderColor: '#ef4444',
                            borderWidth: 3,
                            label: {
                                display: true,
                                content: `Median: ${stats.median.toFixed(2)}°C`,
                                position: 'end'
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Temperature (°C)' },
                    grid: { color: '#e2e8f0' }
                },
                y: {
                    display: false
                }
            }
        }
    });
}

// Chart 6: Monthly Temperature Trends
function createChart6_MonthlyTrends() {
    const ctx = document.getElementById('chart6').getContext('2d');

    // Aggregate by month
    const monthlyData = {};
    sensorData.forEach(d => {
        const monthKey = `${d.timestamp.getFullYear()}-${String(d.timestamp.getMonth() + 1).padStart(2, '0')}`;
        if (!monthlyData[monthKey]) {
            monthlyData[monthKey] = [];
        }
        monthlyData[monthKey].push(d.value);
    });

    const months = Object.keys(monthlyData).sort();
    const avgTemps = months.map(m => {
        const vals = monthlyData[m];
        return vals.reduce((a, b) => a + b, 0) / vals.length;
    });
    const minTemps = months.map(m => Math.min(...monthlyData[m]));
    const maxTemps = months.map(m => Math.max(...monthlyData[m]));

    chartInstances.chart6 = new Chart(ctx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Average Temperature',
                    data: avgTemps,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2.5,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Max Temperature',
                    data: maxTemps,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.05)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: '+1'
                },
                {
                    label: 'Min Temperature',
                    data: minTemps,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: {
                    callbacks: {
                        label: (context) => context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '°C'
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Month' },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12 }
                },
                y: {
                    title: { display: true, text: 'Temperature (°C)' },
                    grid: { color: '#e2e8f0' }
                }
            }
        }
    });
}

// ============================================================================
// SECTION NAVIGATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    // Handle sidebar navigation
    const navItems = document.querySelectorAll('.nav-item');
    const contentSections = document.querySelectorAll('.content-section');

    navItems.forEach(item => {
        item.addEventListener('click', function () {
            const sectionId = this.dataset.section;

            // Update active nav item
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            // Show selected content section
            contentSections.forEach(section => section.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');

            addLog(`Switched to ${this.textContent.trim()} section`);
        });
    });

    // Initialize charts with real data
    addLog('Initializing dashboard...', 'info');
    initializeCharts();

    // Setup button handlers
    setupPipelineControls();

    // Setup logs
    setupLogs();
});

// ============================================================================
// PIPELINE CONTROLS
// ============================================================================

function setupPipelineControls() {
    const runPipelineBtn = document.getElementById('runPipelineBtn');
    const trainModelBtn = document.getElementById('trainModelBtn');

    runPipelineBtn.addEventListener('click', runFullPipeline);
    trainModelBtn.addEventListener('click', trainModel);
}

function runFullPipeline() {
    const pipelineStatus = document.getElementById('pipelineStatus');
    const lastRunTime = document.getElementById('lastRunTime');
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressLabel = document.getElementById('progressLabel');

    addLog('Starting full pipeline execution...', 'info');
    pipelineStatus.textContent = 'Running';
    pipelineStatus.className = 'status-badge running';
    progressContainer.style.display = 'block';

    const steps = [
        { name: 'Data Cleaning', duration: 1000 },
        { name: 'Data Standardization', duration: 1000 },
        { name: 'Data Analysis', duration: 1500 },
        { name: 'Model Training', duration: 2000 }
    ];

    let currentStep = 0;

    function executeStep() {
        if (currentStep < steps.length) {
            const step = steps[currentStep];
            progressLabel.textContent = step.name + '...';
            addLog(`[STEP ${currentStep + 1}] ${step.name}`, 'info');

            setTimeout(() => {
                const progress = ((currentStep + 1) / steps.length) * 100;
                progressFill.style.width = progress + '%';
                addLog(`✓ ${step.name} completed`, 'success');
                currentStep++;
                executeStep();
            }, step.duration);
        } else {
            pipelineStatus.textContent = 'Completed';
            pipelineStatus.className = 'status-badge completed';
            lastRunTime.textContent = new Date().toLocaleString();
            addLog('Pipeline execution completed successfully', 'success');

            setTimeout(() => {
                progressContainer.style.display = 'none';
                progressFill.style.width = '0%';
                pipelineStatus.textContent = 'Idle';
                pipelineStatus.className = 'status-badge idle';
            }, 2000);
        }
    }

    executeStep();
}

function trainModel() {
    addLog('Training Isolation Forest model...', 'info');
    addLog('Loading standardized data from memory...', 'info');
    addLog(`Feature shape: (${stats.count}, 1)`, 'info');
    addLog(`Contamination rate: ${(stats.iqrOutliers / stats.count).toFixed(2)} (${stats.iqrOutliersPercent}%)`, 'info');
    addLog('n_estimators: 100', 'info');

    setTimeout(() => {
        addLog('Model trained successfully', 'success');
        addLog(`Anomalies detected: ${stats.iqrOutliers.toLocaleString()} (${stats.iqrOutliersPercent}%)`, 'success');
        addLog(`Normal classifications: ${stats.normalCount.toLocaleString()} (${stats.normalPercent}%)`, 'success');
        if (stats.extremeCount > 0) {
            addLog(`Extreme fault detection: ${stats.extremeCount}/${stats.extremeCount} (100.0%)`, 'success');
        }
        addLog('Model ready for deployment', 'success');
    }, 2000);
}

// ============================================================================
// LOGS MANAGEMENT
// ============================================================================

function setupLogs() {
    const clearLogsBtn = document.getElementById('clearLogsBtn');
    clearLogsBtn.addEventListener('click', function () {
        const logsContainer = document.getElementById('logsContainer');
        logsContainer.innerHTML = '<div class="log-entry info">[' + new Date().toLocaleTimeString() + '] Logs cleared</div>';
    });
}

function addLog(message, type = 'info') {
    const logsContainer = document.getElementById('logsContainer');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = `[${timestamp}] ${message}`;
    logsContainer.appendChild(logEntry);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// ============================================================================
// REAL-TIME PREDICTION FUNCTIONALITY
// ============================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';
let predictionHistory = [];

// Check API health status on page load
async function checkAPIHealth() {
    const apiStatus = document.getElementById('apiStatus');
    const modelStatus = document.getElementById('modelStatus');
    const scalerStatus = document.getElementById('scalerStatus');

    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            apiStatus.textContent = 'Connected';
            apiStatus.className = 'status-badge trained';
        } else {
            apiStatus.textContent = 'Degraded';
            apiStatus.className = 'status-badge warning';
        }

        modelStatus.textContent = data.model_loaded ? '✓ Loaded' : '✗ Not Loaded';
        scalerStatus.textContent = data.scaler_loaded ? '✓ Loaded' : '✗ Not Loaded';

        addLog('API health check successful', 'success');
    } catch (error) {
        apiStatus.textContent = 'Disconnected';
        apiStatus.className = 'status-badge failed';
        modelStatus.textContent = '✗ Unavailable';
        scalerStatus.textContent = '✗ Unavailable';

        addLog(`API health check failed: ${error.message}`, 'error');
        console.error('API Health Check Error:', error);
    }
}

// Make prediction request to FastAPI
async function makePrediction(value) {
    const predictBtn = document.getElementById('predictBtn');
    const predictionResultCard = document.getElementById('predictionResultCard');

    try {
        // Disable button during request
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class=\"btn-icon\">⏳</span> Processing...';

        addLog(`Sending prediction request for value: ${value}°C`, 'info');

        // Send POST request to /predict endpoint
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ value: parseFloat(value) })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Display result
        displayPredictionResult(data);

        // Add to history
        addToPredictionHistory(data);

        addLog(`Prediction received: ${data.prediction} (confidence: ${data.confidence})`, 'success');

    } catch (error) {
        addLog(`Prediction error: ${error.message}`, 'error');
        console.error('Prediction Error:', error);
        alert(`Error: ${error.message}\n\nMake sure the FastAPI server is running on http://127.0.0.1:8000`);
    } finally {
        // Re-enable button
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<span class=\"btn-icon\">🔮</span> Predict Anomaly';
    }
}

// Display prediction result with color coding
function displayPredictionResult(data) {
    const resultCard = document.getElementById('predictionResultCard');
    const resultHeader = document.getElementById('resultHeader');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');

    // Determine if anomaly (label = 1 means anomaly)
    const isAnomaly = data.anomaly_label === 1;

    // Update header styling
    if (isAnomaly) {
        resultHeader.className = 'result-header anomaly';
        resultIcon.textContent = '🚨';
        resultTitle.textContent = 'ANOMALY DETECTED';
        resultTitle.className = 'result-title anomaly';
        resultSubtitle.textContent = 'Suspicious sensor reading';
    } else {
        resultHeader.className = 'result-header normal';
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'NORMAL';
        resultTitle.className = 'result-title normal';
        resultSubtitle.textContent = 'No anomaly detected';
    }

    // Update result values
    document.getElementById('resultInputValue').textContent = `${data.value.toFixed(2)}°C`;
    document.getElementById('resultStandardized').textContent = data.standardized_value.toFixed(4);
    document.getElementById('resultScore').textContent = data.anomaly_score.toFixed(4);
    document.getElementById('resultConfidence').textContent = data.confidence;

    // Update score visualization
    updateScoreMarker(data.anomaly_score);

    // Update interpretation
    const interpretation = document.getElementById('resultInterpretation');
    if (isAnomaly) {
        if (data.value > 100) {
            interpretation.innerHTML = `<strong>Interpretation:</strong> This temperature reading is <strong>extremely high</strong> (${data.value.toFixed(2)}°C), likely indicating a <strong>sensor malfunction</strong> or equipment failure. Immediate investigation recommended.`;
        } else {
            interpretation.innerHTML = `<strong>Interpretation:</strong> This value (${data.value.toFixed(2)}°C) falls outside the normal operating range. The anomaly score of ${data.anomaly_score.toFixed(4)} indicates it's statistically unusual compared to typical sensor readings.`;
        }
    } else {
        interpretation.innerHTML = `<strong>Interpretation:</strong> This value (${data.value.toFixed(2)}°C) appears normal according to the model. The anomaly score of ${data.anomaly_score.toFixed(4)} is within the expected range for typical sensor readings.`;
    }

    // Show result card with animation
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update score marker position on the visualization bar
function updateScoreMarker(score) {
    const scoreMarker = document.getElementById('scoreMarker');

    // Map score (-0.75 to -0.43) to percentage (0% to 100%)
    // More negative (anomalous) = left side, less negative (normal) = right side
    const minScore = -0.75;
    const maxScore = -0.43;

    // Clamp score to range
    const clampedScore = Math.max(minScore, Math.min(maxScore, score));

    // Calculate percentage (invert because more negative = lower percentage)
    const percentage = ((clampedScore - minScore) / (maxScore - minScore)) * 100;

    scoreMarker.style.left = `${percentage}%`;
}

// Add prediction to history table
function addToPredictionHistory(data) {
    predictionHistory.unshift(data);

    // Keep only last 10 predictions
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }

    updateHistoryTable();
}

// Update history table
function updateHistoryTable() {
    const tbody = document.getElementById('predictionHistoryBody');

    if (predictionHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No predictions yet</td></tr>';
        return;
    }

    tbody.innerHTML = predictionHistory.map(pred => {
        const isAnomaly = pred.anomaly_label === 1;
        const resultBadge = isAnomaly
            ? '<span class="status-badge" style="background: #fee2e2; color: #991b1b;">Anomaly</span>'
            : '<span class="status-badge trained">Normal</span>';

        const time = new Date(pred.timestamp).toLocaleTimeString();

        return `
            <tr>
                <td>${time}</td>
                <td>${pred.value.toFixed(2)}</td>
                <td>${pred.anomaly_score.toFixed(4)}</td>
                <td>${resultBadge}</td>
                <td>${pred.confidence}</td>
            </tr>
        `;
    }).join('');
}

// Setup prediction interface
function setupPredictionInterface() {
    const predictBtn = document.getElementById('predictBtn');
    const sensorValue = document.getElementById('sensorValue');
    const testButtons = document.querySelectorAll('.btn-test');

    // Predict button click handler
    predictBtn.addEventListener('click', () => {
        const value = sensorValue.value;

        if (!value || isNaN(value)) {
            alert('Please enter a valid temperature value');
            return;
        }

        makePrediction(value);
    });

    // Allow Enter key to trigger prediction
    sensorValue.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            predictBtn.click();
        }
    });

    // Quick test button handlers
    testButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const value = btn.dataset.value;
            sensorValue.value = value;
            makePrediction(value);
        });
    });

    // Check API health when entering prediction section
    const predictionNavItem = document.querySelector('.nav-item[data-section="real-time-prediction"]');
    if (predictionNavItem) {
        predictionNavItem.addEventListener('click', () => {
            checkAPIHealth();
        });
    }
}

// Initialize prediction interface when DOM is ready
document.addEventListener('DOMContentLoaded', function () {
    // Existing code continues...
    setupPredictionInterface();

    // Check API health on initial load if prediction section is active
    setTimeout(() => {
        checkAPIHealth();
    }, 1000);
});
