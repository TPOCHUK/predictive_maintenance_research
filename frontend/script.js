const $ = id => document.getElementById(id);
const engineSelect = $("engine-select");
const sensorSelect = $("sensor-select");
const charts = new Map();
let busy = false;
let refreshing = false;
let state = {status: "idle"};
let lastData;
const sensorColors = ["#0369a1", "#7c3aed", "#0f766e", "#c2410c", "#be185d",
    "#4338ca", "#15803d", "#a16207", "#0e7490", "#a21caf", "#b45309",
    "#047857", "#6d28d9", "#9f1239", "#1d4ed8", "#4d7c0f", "#b91c1c",
    "#115e59", "#86198f", "#92400e", "#334155"];
const sensorColor = name => sensorColors[Number(name.split("_")[1]) - 1];
const formatValue = value => Number(value).toLocaleString("uk-UA", {maximumFractionDigits: 5});

function rollingMean(values, window = 5) {
    return values.map((_, i) => i < window - 1 ? null
        : values.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0) / window);
}

async function api(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

// Draw a vertical line where a sensor change was confirmed.
const eventMarkers = {
    id: "eventMarkers",
    afterDraw(chart) {
        const events = chart.options.plugins.eventMarkers?.events || [];
        const {ctx, chartArea, scales} = chart;
        if (!chartArea) return;
        ctx.save();
        ctx.font = "11px Arial";
        events.forEach((event, index) => {
            const x = scales.x.getPixelForValue(event.cycle);
            if (x < chartArea.left || x > chartArea.right) return;
            ctx.strokeStyle = "#dc2626";
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
            if (index < 3) {
                ctx.fillStyle = "#b91c1c";
                ctx.textAlign = x > (chartArea.left + chartArea.right) / 2 ? "right" : "left";
                ctx.fillText(`${event.sensor} · цикл ${event.cycle}`, x, chartArea.top + 12 + index * 14);
            }
        });
        ctx.restore();
    }
};

function line(id, label, cycles, values, color, options = {}) {
    const {events = [], threshold = null, comparison = null} = options;
    if (typeof Chart === "undefined") throw new Error("Не вдалося завантажити Chart.js. Перевір інтернет і онови сторінку.");
    const isSensor = id === "sensor-chart";
    const rollingMeanCache = isSensor ? rollingMean(values) : [];
    const datasets = [{label: isSensor ? "Вимірювання" : label, data: cycles.map((x, i) => ({x, y: values[i]})),
        borderColor: isSensor ? `${color}88` : color, borderWidth: isSensor ? 1.5 : 2.5,
        pointRadius: isSensor ? 0 : 1, pointHoverRadius: 4, tension: 0, fill: false}];
    if (isSensor) datasets.push({label: "Тренд · середнє за 5 циклів",
        data: cycles.map((x, i) => ({x, y: rollingMeanCache[i]})),
        borderColor: color, borderWidth: 3, pointRadius: 0, tension: 0.15, fill: false});
    if (comparison) datasets.push({label: "Еталонний RUL · FD001", borderColor: "#7c3aed",
        data: comparison.cycles.map((x, i) => ({x, y: comparison.values[i]})),
        pointRadius: 0, borderDash: [7, 4], borderWidth: 2.5, fill: false});
    if (threshold !== null) datasets.push({label: "Поріг 3σ", borderColor: "#dc2626",
        data: cycles.map(x => ({x, y: threshold})), pointRadius: 0, borderDash: [5, 5], borderWidth: 1});
    const existing = charts.get(id);
    if (existing) {
        existing.data.datasets = datasets;
        existing.options.plugins.eventMarkers.events = events;
        existing.update("none");
        return;
    }
    charts.set(id, new Chart($(id), {
        type: "line", plugins: [eventMarkers], data: {datasets},
        options: {responsive: true, maintainAspectRatio: false, animation: false,
            devicePixelRatio: Math.min(3, Math.max(2, globalThis.devicePixelRatio || 1)),
            interaction: {mode: "index", intersect: false},
            plugins: {legend: {display: true, labels: {usePointStyle: true}}, eventMarkers: {events},
                tooltip: {callbacks: {label: item => `${item.dataset.label}: ${formatValue(item.parsed.y)}`}}},
            scales: {x: {type: "linear", display: true, title: {display: true, text: "Цикл"},
                grid: {display: true, color: "#e2e8f0"}, ticks: {precision: 0, maxTicksLimit: 10, color: "#475569"}},
                y: {display: true, beginAtZero: false, grace: "8%", grid: {color: "#e2e8f0"},
                    ticks: {maxTicksLimit: 7, color: "#475569", callback: formatValue},
                    title: {display: true, text: label}}}}
    }));
}

function controls() {
    const running = state.status === "running";
    $("start-button").disabled = busy || running || !engineSelect.value;
    $("stop-button").disabled = busy || !running;
    $("predict-button").disabled = busy || !lastData?.current_cycle;
    engineSelect.disabled = busy || running;
    $("speed-select").disabled = busy || running;
}

function render(data) {
    lastData = data;
    state = data.simulation;
    $("current-cycle").textContent = data.current_cycle;
    $("rul-engine-label").textContent = `Engine #${data.unit_id}`;
    const prediction = data.latest_prediction;
    $("predicted-rul").textContent = prediction ? prediction.predicted_rul.toFixed(2) : "—";
    $("reference-rul").textContent = data.reference_rul ?? "—";
    $("prediction-comparison").textContent = data.prediction_error != null
        ? `На циклі ${prediction.cycle}: еталон ${data.prediction_reference_rul}, різниця ${data.prediction_error >= 0 ? "+" : ""}${data.prediction_error.toFixed(2)} циклів. Еталон вище — на поточному циклі.`
        : "Еталон із FD001 · лише для демонстрації";
    $("engine-status").textContent = prediction ? (prediction.is_warmup ? "Warmup · <80 циклів" : "Monitoring") : "Очікування прогнозу";
    const labels = {idle: "Готово до запуску", running: "Симуляція працює", stopped: "Зупинено", completed: "Завершено", error: "Помилка симуляції"};
    $("simulation-status").textContent = `${labels[state.status] || state.status}${state.unit_id ? ` · двигун ${state.unit_id} · ${state.cycle}/${state.total_cycles} циклів` : ""}`;
    $("simulation-progress").value = state.total_cycles ? state.cycle / state.total_cycles * 100 : 0;
    if (state.error) $("request-error").textContent = state.error;

    const overview = data.overview;
    const events = overview.events;
    const first = events[0];
    $("anomaly-summary").textContent = !overview.ready
        ? `Збираємо початкову базу: ${Math.min(data.current_cycle, 20)}/20 циклів.`
        : first ? `Перша підтверджена аномалія: цикл ${first.cycle}, ${first.sensor}, відхилення ${first.score}σ.`
            : "Стійких відхилень понад поріг поки не виявлено.";
    line("rul-chart", "Прогноз RUL · цикли", data.prediction_cycles, data.predicted_rul_values, "#16a34a",
        {events: events.slice(0, 1), comparison: {cycles: data.telemetry_cycles, values: data.reference_rul_values || []}});
    line("overview-chart", "Максимальне відхилення · σ", overview.cycles, overview.scores, "#2563eb", {events, threshold: 3});
    renderSensor(data);
    const eventList = $("anomaly-events");
    eventList.replaceChildren();
    events.forEach(event => {
        const button = document.createElement("button");
        button.textContent = `${event.sensor} · цикл ${event.cycle} · ${event.score}σ`;
        button.onclick = () => selectSensor(event.sensor);
        eventList.append(button);
    });
    controls();
}

function renderSensor(data) {
    const name = sensorSelect.value;
    line("sensor-chart", name, data.telemetry_cycles, data.sensor_series[name] || [], sensorColor(name),
        {events: data.overview.events.filter(event => event.sensor === name)});
}

function selectSensor(name) {
    sensorSelect.value = name;
    if (lastData) renderSensor(lastData);
    $("sensor-chart").scrollIntoView({behavior: "smooth", block: "center"});
}

// Poll the API once a second. Skip overlapping requests.
async function refresh() {
    if (refreshing || !engineSelect.value) return;
    refreshing = true;
    const id = engineSelect.value;
    try {
        const data = await api(`/engines/${id}/dashboard`);
        if (id === engineSelect.value) render(data);
        $("connection-status").textContent = "API online · оновлення щосекунди";
    } catch (error) {
        $("connection-status").textContent = "API / графіки недоступні";
        $("request-error").textContent = error.message;
    } finally { refreshing = false; }
}

async function action(url, body) {
    busy = true; controls(); $("request-error").textContent = "";
    try {
        await api(url, {method: "POST", headers: {"Content-Type": "application/json"},
            ...(body ? {body: JSON.stringify(body)} : {})});
        await refresh();
    } catch (error) { $("request-error").textContent = error.message; }
    finally { busy = false; controls(); }
}

async function init() {
    sensorSelect.replaceChildren();
    for (let i = 1; i <= 21; i++) {
        const name = `sensor_${i}`;
        sensorSelect.add(new Option(`Sensor ${i}`, name));
    }
    sensorSelect.value = "sensor_4";
    const [catalog, current] = await Promise.all([api("/engines"), api("/simulation")]);
    state = current;
    catalog.engines.forEach(id => engineSelect.add(new Option(`Engine #${id}`, id)));
    if (current.unit_id) engineSelect.value = current.unit_id;
    controls();
    engineSelect.onchange = () => {lastData = null; controls(); refresh();};
    sensorSelect.onchange = () => lastData && renderSensor(lastData);
    $("start-button").onclick = () => action("/simulation/start", {unit_id: Number(engineSelect.value), interval: Number($("speed-select").value)});
    $("stop-button").onclick = () => action("/simulation/stop");
    $("predict-button").onclick = () => action(`/engines/${engineSelect.value}/predict`);
    await refresh();
    setInterval(refresh, 1000);
}

init().catch(error => { $("request-error").textContent = error.message; });
