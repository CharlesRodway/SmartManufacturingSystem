// dashboard.js
// Shared core logic — polling, state management, event log, alerts
// Used by both dashboard.html (supervisor) and maintenance.html

// ── Machine registry ──────────────────────────────────────────────────────────
// Machine type modules register themselves here.
// To add a new machine type: create machines/newtype.js and call
// MachineRegistry.register('newtype', NewTypeModule) at the bottom of that file.

const MachineRegistry = {
    _modules: {},
    register(type, module) {
        this._modules[type] = module;
        console.log(`MachineRegistry: registered type "${type}"`);
    },
    get(type) {
        return this._modules[type] || null;
    },
    types() {
        return Object.keys(this._modules);
    }
};

// ── Config ────────────────────────────────────────────────────────────────────

const BACKEND    = 'http://localhost:8000';
const POLL_MS    = 2000;
const STREAK_MAX = 50;
const HISTORY_N  = 200;
const SPARK_N    = 30;
const LOG_MAX    = 80;
const B_COLOURS  = ['#1a7a3c', '#1f6feb', '#e08000', '#7c3aed'];

// ── Shared state ──────────────────────────────────────────────────────────────

let systemState    = {};   // bearing machines
let prevStatuses   = {};
let eventLog       = [];

// ── API calls ─────────────────────────────────────────────────────────────────

async function fetchState() {
    try {
        const r    = await fetch(`${BACKEND}/state`);
        if (!r.ok) throw new Error();
        updateBackendPill(true);
        return await r.json();
    } catch {
        updateBackendPill(false);
        return null;
    }
}

async function fetchHistory(lathe) {
    try {
        const r = await fetch(`${BACKEND}/history/${lathe}?n=${HISTORY_N}`);
        if (!r.ok) return [];
        return await r.json();
    } catch { return []; }
}

async function fetchAlerts() {
    try {
        const r = await fetch(`${BACKEND}/alerts`);
        if (!r.ok) return [];
        return await r.json();
    } catch { return []; }
}

// ── Backend status pill ───────────────────────────────────────────────────────

function updateBackendPill(ok) {
    const pill = document.getElementById('backend-pill');
    if (!pill) return;
    pill.className   = ok ? 'pill ok' : 'pill err';
    pill.textContent = ok ? 'BACKEND OK' : 'OFFLINE';
}

// ── Spark bars ────────────────────────────────────────────────────────────────

function makeSparkBars(scores) {
    if (!scores || scores.length === 0) return '';
    const maxAbs = Math.max(...scores.map(v => Math.abs(v)), 0.001);
    return scores.map(v => {
        const h  = Math.max(2, Math.round((Math.abs(v) / maxAbs) * 18));
        const bg = v < 0 ? '#ba1b1b' : '#1a7a3c';
        return `<div class="spark-bar" style="height:${h}px;background:${bg};opacity:0.65"></div>`;
    }).join('');
}

// ── Event log ─────────────────────────────────────────────────────────────────

function addLog(level, text) {
    eventLog.unshift({ time: new Date().toTimeString().slice(0, 8), level, text });
    if (eventLog.length > LOG_MAX) eventLog.length = LOG_MAX;
}

function renderLog() {
    const el = document.getElementById('event-log');
    if (!el) return;
    const countEl = document.getElementById('event-count');
    if (countEl) countEl.textContent = eventLog.length;
    el.innerHTML = eventLog.length === 0
        ? '<div class="no-data-msg">Waiting for events...</div>'
        : eventLog.map(e =>
            `<div class="log-entry ${e.level}">
                <span class="log-time">${e.time}</span>
                <span>${e.text}</span>
            </div>`).join('');
}

function detectChanges(newState) {
    for (const [lathe, data] of Object.entries(newState)) {
        const display = lathe.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        for (const [bearing, bdata] of Object.entries(data.bearings || {})) {
            const key  = `${lathe}_${bearing}`;
            const prev = prevStatuses[key];
            const curr = bdata.status;
            const bl   = bearing.replace('bearing', 'B');
            if (prev && prev !== curr) {
                if      (curr === 'CRITICAL') addLog('critical', `${display} ${bl} CRITICAL (streak: ${bdata.streak})`);
                else if (curr === 'WARNING')  addLog('warning',  `${display} ${bl} WARNING (streak: ${bdata.streak})`);
                else if (curr === 'HEALTHY')  addLog('healthy',  `${display} ${bl} returned HEALTHY`);
            }
            prevStatuses[key] = curr;
        }
    }
}

// ── Alerts ────────────────────────────────────────────────────────────────────

function renderAlerts(alerts) {
    const countEl = document.getElementById('alert-count');
    const listEl  = document.getElementById('alert-list');
    if (countEl) countEl.textContent = alerts.length;
    if (!listEl) return;
    if (alerts.length === 0) {
        listEl.innerHTML = '<div class="no-data-msg">No critical alerts logged.</div>';
        return;
    }
    listEl.innerHTML = alerts.map(a => {
        const machine = a.lathe.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
        const bearing = a.bearing.replace('bearing','Bearing ');
        return `<div class="alert-item">
            <div class="ai-title">${machine} — ${bearing}</div>
            <div class="ai-meta">
                ${a.timestamp?.slice(0,19).replace('T',' ') || 'N/A'}<br>
                Score: ${a.score?.toFixed(4)} · Kurt: ${a.kurtosis?.toFixed(3)}<br>
                Reading: ${a.reading}
            </div>
        </div>`;
    }).join('');
}

// ── Heatmap ───────────────────────────────────────────────────────────────────

function renderHeatmap(bearingState, hydState) {
    const grid   = document.getElementById('heatmap-grid');
    if (!grid) return;
    const lathes = Object.keys(bearingState);
    if (lathes.length === 0) { grid.innerHTML = ''; return; }

    let html = '';

    if (lathes.length > 0) {
        const bearings = Object.keys(bearingState[lathes[0]]?.bearings || {});
        html += `<div class="hm-row">
            <div class="hm-lathe-label"></div>
            ${bearings.map(b => `<div class="hm-cell" style="background:none;border:none;font-size:0.55rem;color:var(--muted);height:12px;">${b.replace('bearing','B')}</div>`).join('')}
        </div>`;
        const mod = MachineRegistry.get('lathe');
        lathes.forEach(lathe => {
            if (mod) html += mod.renderHeatmapRow(lathe, bearingState[lathe], bearings);
        });
    }

    grid.innerHTML = html;
}

// ── Clock ─────────────────────────────────────────────────────────────────────

function startClock() {
    const el = document.getElementById('clock');
    if (!el) return;
    setInterval(() => {
        el.textContent = new Date().toTimeString().slice(0, 8);
    }, 1000);
}
