// Browser logic smoke test without external browser or chart dependencies.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map();
const calls = [];
const chartConfigs = new Map();
let state = {status: 'idle', unit_id: null, cycle: 0, total_cycles: 40};
function element(id) {
    if (!elements.has(id)) elements.set(id, {
        id, setAttribute() {},
        value: '', children: [], classList: {toggle() {}},
        add(option) {this.children.push(option); if (!this.value) this.value = String(option.value);},
        append(child) {this.children.push(child);}, replaceChildren() {this.children = [];},
        scrollIntoView() {}
    });
    return elements.get(id);
}
const context = {
    document: {getElementById: element, createElement: () => element(`temp-${elements.size}`)},
    Option: function(text, value) {this.text = text; this.value = value;},
    Chart: function(canvas, config) {this.data = config.data; this.options = config.options; this.update = () => {}; chartConfigs.set(canvas.id, this);},
    setInterval() {}, console,
    fetch: async (url, options = {}) => {
        calls.push([url, options]);
        if (url === '/simulation/start') state = {status: 'running', unit_id: 1, cycle: 5, total_cycles: 40};
        if (url === '/simulation/stop') state.status = 'stopped';
        let body = state;
        if (url === '/engines') body = {engines: Array.from({length: 100}, (_, i) => i + 1)};
        if (url.includes('/dashboard')) body = {
            unit_id: 1, current_cycle: state.cycle, simulation: {...state},
            latest_prediction: null, prediction_cycles: [], predicted_rul_values: [],
            reference_rul: 143 - state.cycle, reference_rul_values: [], prediction_error: null,
            telemetry_cycles: [], sensor_series: Object.fromEntries(Array.from({length: 21}, (_, i) => [`sensor_${i + 1}`, []])),
            overview: {ready: false, events: [], cycles: [], scores: [], sensors: []}
        };
        return {ok: true, json: async () => body};
    }
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('frontend/script.js', 'utf8'), context);
setImmediate(async () => {
    try {
        assert.equal(element('engine-select').children.length, 100);
        assert.equal(element('sensor-select').children.length, 21);
        element('speed-select').value = '0.5';
        await element('start-button').onclick();
        assert.equal(element('start-button').disabled, true);
        assert.equal(element('stop-button').disabled, false);
        assert.equal(element('current-cycle').textContent, 5);
        assert.equal(element('reference-rul').textContent, 138);
        assert.equal(chartConfigs.get('rul-chart').data.datasets[1].label, 'Еталонний RUL · FD001');
        assert.equal(chartConfigs.get('sensor-chart').data.datasets.length, 2);
        assert.equal(chartConfigs.get('sensor-chart').options.scales.y.display, true);
        assert.equal(chartConfigs.size, 3);
        await element('stop-button').onclick();
        assert.equal(element('start-button').disabled, false);
        assert.equal(element('stop-button').disabled, true);
        assert(calls.some(([url]) => url === '/simulation/start'));
        console.log('Frontend simulation controls: OK');
    } catch (error) { console.error(error); process.exitCode = 1; }
});
