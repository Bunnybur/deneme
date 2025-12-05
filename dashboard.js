
const API_BASE_URL = 'http://localhost:8000';

let sensorData = [];
let stats = {};
let chartInstances = {};
let predictionHistory = [];

async function loadSensorDataFromAPI() {
    try {
        addLog('Loading sensor readings from FastAPI...', 'info');

        const response = await fetch(`${API_BASE_URL}/readings`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const readings = await response.json();

        sensorData = readings.map(reading => ({
            id: reading.id,
            timestamp: new Date(reading.timestamp),
            value: parseFloat(reading.value),
            status: reading.status,
            confidence_score: reading.confidence_score
        }));

        sensorData.sort((a, b) => a.timestamp - b.timestamp);

        addLog(`Loaded ${sensorData.length} sensor readings from API`, 'success');
        calculateStatistics();
        return sensorData;

    } catch (error) {
        addLog(`Error loading data from API: ${error.message}`, 'error');
        addLog('Make sure FastAPI server is running on http://localhost:8000', 'error');
        console.error('API Error:', error);
        throw error;
    }
}

async function loadStatsFromAPI() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const apiStats = await response.json();

        document.getElementById('totalRecords').textContent = apiStats.total_readings.toLocaleString();
        document.getElementById('normalPercent').textContent =
            ((apiStats.normal_count / apiStats.total_readings) * 100).toFixed(1) + '%';
        document.getElementById('extremeFaults').textContent =
            apiStats.fault_percentage.toFixed(2) + '%';

        addLog('Stats loaded from API', 'success');
        return apiStats;

    } catch (error) {
        addLog(`Error loading stats: ${error.message}`, 'error');
        console.error('Stats Error:', error);
    }
}

function calculateStatistics() {
    const values = sensorData.map(d => d.value);
    const n = values.length;

    const mean = values.reduce((a, b) => a + b, 0) / n;
    const sorted = [...values].sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[n - 1];
    const q1 = sorted[Math.floor(n * 0.25)];
    const median = sorted[Math.floor(n * 0.5)];
    const q3 = sorted[Math.floor(n * 0.75)];

    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);

    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

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
    document.getElementById('iqrOutliers').textContent = stats.iqrOutliersPercent + '%';

    const firstDate = sensorData[0].timestamp;
    const lastDate = sensorData[sensorData.length - 1].timestamp;
    const monthsDiff = Math.round((lastDate - firstDate) / (1000 * 60 * 60 * 24 * 30));

    document.getElementById('dateRange').innerHTML = `
        <div>${firstDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</div>
        <div>→</div>
        <div>${lastDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</div>
    `;
    document.querySelector('#dateRange').nextElementSibling.textContent = `${monthsDiff} Months`;

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

    document.querySelectorAll('.iqr-value')[0].textContent = stats.q1.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[1].textContent = stats.q3.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[2].textContent = stats.iqr.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[3].textContent = stats.lowerBound.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[4].textContent = stats.upperBound.toFixed(2) + '°C';
    document.querySelectorAll('.iqr-value')[5].textContent = `${stats.iqrOutliers.toLocaleString()} (${stats.iqrOutliersPercent}%)`;

    document.getElementById('tempRange').textContent = `${stats.min.toFixed(2)}°C - ${stats.max.toFixed(2)}°C`;
    document.getElementById('meanTemp').textContent = `${stats.mean.toFixed(2)}°C`;
    document.getElementById('stdDev').textContent = `${stats.std.toFixed(2)}°C`;
}

async function initializeCharts() {
    try {
        await loadSensorDataFromAPI();
        await loadStatsFromAPI();

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

function createChart1_CompleteTimeSeries() {
    const ctx = document.getElementById('chart1').getContext('2d');

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

function createChart4_Distribution() {
    const ctx = document.getElementById('chart4').getContext('2d');

    const binSize = 5;
    const bins = {};
    sensorData.forEach(d => {
        const bin = Math.floor(d.value / binSize) * binSize;
        bins[bin] = (bins[bin] || 0) + 1;
    });

    const sortedBins = Object.keys(bins).map(Number).sort((a, b) => a - b);
    const colors = sortedBins.map(bin => {
        if (bin > 100) return '#ef4444';
        if (bin < stats.lowerBound || bin > stats.upperBound) return '#f59e0b';
        return '#3b82f6';
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

function createChart5_BoxPlot() {
    const ctx = document.getElementById('chart5').getContext('2d');

    const values = sensorData.map(d => d.value);
    const sorted = [...values].sort((a, b) => a - b);
    const outlierPoints = values.filter(v => v < stats.lowerBound || v > stats.upperBound);

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

function createChart6_MonthlyTrends() {
    const ctx = document.getElementById('chart6').getContext('2d');

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

document.addEventListener('DOMContentLoaded', function () {
    const navItems = document.querySelectorAll('.nav-item');
    const contentSections = document.querySelectorAll('.content-section');

    navItems.forEach(item => {
        item.addEventListener('click', function () {
            const sectionId = this.dataset.section;

            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            contentSections.forEach(section => section.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');

            addLog(`Switched to ${this.textContent.trim()} section`);
        });
    });

    addLog('Initializing dashboard...', 'info');
    initializeCharts();

    setupPipelineControls();
    setupLogs();
    setupPredictionInterface();

    setTimeout(() => {
        checkAPIHealth();
    }, 1000);
});

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
    addLog('Training supervised ML model...', 'info');
    addLog('Loading standardized data from API...', 'info');
    addLog(`Feature shape: (${stats.count}, 1)`, 'info');
    addLog('Training Random Forest, Gradient Boosting, XGBoost...', 'info');

    setTimeout(() => {
        addLog('Model trained successfully', 'success');
        addLog(`Normal classifications: ${stats.normalCount.toLocaleString()} (${stats.normalPercent}%)`, 'success');
        if (stats.extremeCount > 0) {
            addLog(`Fault detection: ${stats.extremeCount} faults identified`, 'success');
        }
        addLog('Model ready for deployment', 'success');
    }, 2000);
}

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

async function checkAPIHealth() {
    const apiStatus = document.getElementById('apiStatus');
    const modelStatus = document.getElementById('modelStatus');
    const scalerStatus = document.getElementById('scalerStatus');

    try {
        const response = await fetch(`${API_BASE_URL}/`);
        const data = await response.json();

        if (response.ok) {
            apiStatus.textContent = 'Connected';
            apiStatus.className = 'status-badge trained';
            modelStatus.textContent = data.status.includes('loaded') ? '✓ Loaded' : '✗ Not Loaded';
            scalerStatus.textContent = '✓ Ready';
        } else {
            apiStatus.textContent = 'Degraded';
            apiStatus.className = 'status-badge warning';
        }

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

async function makePrediction(value) {
    const predictBtn = document.getElementById('predictBtn');
    const predictionResultCard = document.getElementById('predictionResultCard');

    try {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="btn-icon">⏳</span> Processing...';

        addLog(`Sending prediction request for value: ${value}°C`, 'info');

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

        displayPredictionResult(data);
        addToPredictionHistory(data);

        addLog(`Prediction: ${data.status} (confidence: ${data.confidence_score.toFixed(2)})`, 'success');

    } catch (error) {
        addLog(`Prediction error: ${error.message}`, 'error');
        console.error('Prediction Error:', error);
        alert(`Error: ${error.message}\n\nMake sure the FastAPI server is running on http://localhost:8000`);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<span class="btn-icon">🔮</span> Predict Fault';
    }
}

function displayPredictionResult(data) {
    const resultCard = document.getElementById('predictionResultCard');
    const resultHeader = document.getElementById('resultHeader');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');

    const isFault = data.status === "Fault";

    if (isFault) {
        resultHeader.className = 'result-header anomaly';
        resultIcon.textContent = '🚨';
        resultTitle.textContent = 'FAULT DETECTED';
        resultTitle.className = 'result-title anomaly';
        resultSubtitle.textContent = 'Suspicious sensor reading';
    } else {
        resultHeader.className = 'result-header normal';
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'NORMAL';
        resultTitle.className = 'result-title normal';
        resultSubtitle.textContent = 'No fault detected';
    }

    document.getElementById('resultInputValue').textContent = `${data.value.toFixed(2)}°C`;
    document.getElementById('resultStandardized').textContent = (data.value / stats.std).toFixed(4);
    document.getElementById('resultScore').textContent = data.confidence_score.toFixed(4);
    document.getElementById('resultConfidence').textContent = (data.confidence_score * 100).toFixed(1) + '%';

    updateScoreMarker(data.confidence_score);

    const interpretation = document.getElementById('resultInterpretation');
    if (isFault) {
        if (data.value > 100) {
            interpretation.innerHTML = `<strong>Interpretation:</strong> This temperature reading is <strong>extremely high</strong> (${data.value.toFixed(2)}°C), likely indicating a <strong>sensor malfunction</strong> or equipment failure. Immediate investigation recommended.`;
        } else {
            interpretation.innerHTML = `<strong>Interpretation:</strong> This value (${data.value.toFixed(2)}°C) falls outside the normal operating range. The confidence score of ${data.confidence_score.toFixed(4)} indicates it's statistically unusual.`;
        }
    } else {
        interpretation.innerHTML = `<strong>Interpretation:</strong> This value (${data.value.toFixed(2)}°C) appears normal according to the model. The confidence score of ${data.confidence_score.toFixed(4)} is within the expected range.`;
    }

    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateScoreMarker(score) {
    const scoreMarker = document.getElementById('scoreMarker');

    const minScore = 0.0;
    const maxScore = 1.0;

    const clampedScore = Math.max(minScore, Math.min(maxScore, score));
    const percentage = (clampedScore / maxScore) * 100;

    scoreMarker.style.left = `${percentage}%`;
}

function addToPredictionHistory(data) {
    predictionHistory.unshift({
        ...data,
        timestamp: new Date().toISOString()
    });

    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }

    updateHistoryTable();
}

function updateHistoryTable() {
    const tbody = document.getElementById('predictionHistoryBody');

    if (predictionHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No predictions yet</td></tr>';
        return;
    }

    tbody.innerHTML = predictionHistory.map(pred => {
        const isFault = pred.status === "Fault";
        const resultBadge = isFault
            ? '<span class="status-badge" style="background: #fee2e2; color: #991b1b;">Fault</span>'
            : '<span class="status-badge trained">Normal</span>';

        const time = new Date(pred.timestamp).toLocaleTimeString();

        return `
            <tr>
                <td>${time}</td>
                <td>${pred.value.toFixed(2)}</td>
                <td>${pred.confidence_score.toFixed(4)}</td>
                <td>${resultBadge}</td>
                <td>${(pred.confidence_score * 100).toFixed(1)}%</td>
            </tr>
        `;
    }).join('');
}

function setupPredictionInterface() {
    const predictBtn = document.getElementById('predictBtn');
    const sensorValue = document.getElementById('sensorValue');
    const testButtons = document.querySelectorAll('.btn-test');

    predictBtn.addEventListener('click', () => {
        const value = sensorValue.value;

        if (!value || isNaN(value)) {
            alert('Please enter a valid temperature value');
            return;
        }

        makePrediction(value);
    });

    sensorValue.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            predictBtn.click();
        }
    });

    testButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const value = btn.dataset.value;
            sensorValue.value = value;
            makePrediction(value);
        });
    });

    const predictionNavItem = document.querySelector('.nav-item[data-section="real-time-prediction"]');
    if (predictionNavItem) {
        predictionNavItem.addEventListener('click', () => {
            checkAPIHealth();
        });
    }
}
