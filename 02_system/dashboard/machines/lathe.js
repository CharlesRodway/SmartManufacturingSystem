// machines/lathe.js
// Bearing / CNC Lathe machine type definition
// Handles rendering of bearing cards for the supervisor dashboard
//
// To add a new machine type in the future:
//   1. Create a new file in machines/ following this same structure
//   2. Register it with MachineRegistry.register() at the bottom
//   3. Add it to the dashboard.html script imports

const LatheModule = {

    // what to show in the machine list sidebar
    renderListItem(name, state, isSelected) {
        const status  = (state?.machine_status || 'NO DATA').toLowerCase().replace(' ', '');
        const reading = state?.reading || 0;
        const total   = state?.total   || 0;
        const pct     = total > 0 ? (reading / total * 100) : 0;
        const barCol  = status === 'critical' ? 'var(--critical)' : status === 'warning' ? '#e08000' : 'var(--healthy)';
        const display = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const sel     = isSelected ? 'selected' : '';

        return `<div class="mc ${status} ${sel}" onclick="selectMachine('${name}','lathe')">
            <div class="mc-type">CNC Lathe</div>
            <div class="mc-name">${display}</div>
            <div class="mc-status ${status}">${state?.machine_status || 'NO DATA'}</div>
            <div class="mc-meta">${reading} / ${total} readings</div>
            <div class="mc-bar-bg"><div class="mc-bar" style="width:${pct.toFixed(1)}%;background:${barCol}"></div></div>
        </div>`;
    },

    // what to show in the heatmap
    renderHeatmapRow(name, state, bearingNames) {
        const label = name.replace('lathe_', 'L');
        const cells = bearingNames.map(b => {
            const status = state?.bearings?.[b]?.status?.toLowerCase() || 'no-data';
            const short  = status === 'no-data' ? '—' : status.slice(0, 3).toUpperCase();
            return `<div class="hm-cell ${status}" title="${name} ${b}: ${status}">${short}</div>`;
        }).join('');
        return `<div class="hm-row"><div class="hm-lathe-label">${label}</div>${cells}</div>`;
    },

    // the full detail view shown in the centre panel when this machine is selected
    renderDetail(name, state, history, SPARK_N, STREAK_MAX) {
        if (!state) return '';

        const status  = state.machine_status || 'NO DATA';
        const reading = state.reading || 0;
        const total   = state.total   || 0;
        const pct     = total > 0 ? (reading / total * 100) : 0;
        const barCol  = status === 'CRITICAL' ? 'var(--critical)' : status === 'WARNING' ? '#e08000' : 'var(--healthy)';
        const display = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

        // update the shared header elements
        document.getElementById('dh-name').textContent             = display;
        document.getElementById('dh-sub').textContent              = state.filename || '—';
        document.getElementById('detail-progress-txt').textContent = `Reading ${reading} / ${total} (${pct.toFixed(1)}%)`;
        document.getElementById('detail-progress-bar').style.cssText = `width:${pct.toFixed(1)}%;background:${barCol}`;

        const badge       = document.getElementById('dh-badge');
        badge.className   = `status-badge ${status.toLowerCase()}`;
        badge.textContent = status;

        // show bearing grid, hide hydraulic panels
        document.getElementById('bearing-grid').style.display    = 'grid';
        document.getElementById('hydraulic-grid').style.display  = 'none';
        document.getElementById('hyd-sensors').style.display     = 'none';

        const bearings = state.bearings || {};
        document.getElementById('bearing-grid').innerHTML = Object.entries(bearings).map(([bname, b]) => {
            const bStatus     = (b.status || 'HEALTHY').toLowerCase();
            const sparkScores = history.slice(-SPARK_N).map(h => h.bearings?.[bname]?.score).filter(v => v !== undefined);
            const anomalyRate = sparkScores.length > 0
                ? Math.round((sparkScores.filter(v => v < 0).length / sparkScores.length) * 100) : 0;

            const scoreClass  = b.score < 0 ? 'neg' : 'pos';
            const streakClass = b.streak > 30 ? 'neg' : b.streak > 10 ? 'warn' : 'pos';
            const latchedTag  = b.latched ? '<span class="tag latched">LATCHED</span>' : '';

            return `<div class="bc ${bStatus}">
                <div class="bc-top">
                    <div class="bc-name">${bname.replace('bearing', 'Bearing ')}</div>
                    <div class="bc-tags">${latchedTag}</div>
                </div>
                <div class="bc-metrics">
                    <div class="bc-metric">
                        <label>Anomaly Score</label>
                        <div class="val ${scoreClass}">${b.score >= 0 ? '+' : ''}${b.score?.toFixed(4)}</div>
                    </div>
                    <div class="bc-metric">
                        <label>Streak</label>
                        <div class="val ${streakClass}">${b.streak} / ${STREAK_MAX}</div>
                    </div>
                    <div class="bc-metric">
                        <label>Kurtosis</label>
                        <div class="val">${b.kurtosis?.toFixed(3)}</div>
                    </div>
                    <div class="bc-metric">
                        <label>RMS Energy</label>
                        <div class="val">${b.rms?.toFixed(4)}</div>
                    </div>
                </div>
                <div class="spark-bar-wrap">
                    <div class="spark-label">
                        <span>Anomaly rate (last ${SPARK_N})</span>
                        <span>${anomalyRate}%</span>
                    </div>
                    <div class="spark-bars">${makeSparkBars(sparkScores)}</div>
                </div>
            </div>`;
        }).join('');
    },

    // history chart data extraction for this machine type
    getHistoryDatasets(history, B_COLOURS) {
        if (!history || history.length === 0) return { labels: [], score: [], kurtosis: [], rms: [] };
        const bearings = Object.keys(history[0]?.bearings || {});
        const labels   = history.map(h => h.reading);
        const make     = key => bearings.map((b, i) => ({
            label:           b.replace('bearing', 'B'),
            data:            history.map(h => h.bearings[b]?.[key]),
            borderColor:     B_COLOURS[i % B_COLOURS.length],
            backgroundColor: 'transparent',
            borderWidth:     1.5,
            pointRadius:     0,
            tension:         0.3
        }));
        return { labels, score: make('score'), kurtosis: make('kurtosis'), rms: make('rms') };
    },

    // status summary for the maintenance panel
    getStatusSummary(name, state) {
        if (!state) return null;
        const bearings = state.bearings || {};
        const issues   = Object.entries(bearings)
            .filter(([, b]) => b.status !== 'HEALTHY')
            .map(([bname, b]) => `${bname.replace('bearing', 'B')}: ${b.status} (streak ${b.streak})`);
        return {
            name:    name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            type:    'CNC Lathe',
            status:  state.machine_status || 'NO DATA',
            issues,
            reading: `${state.reading || 0} / ${state.total || 0} readings`
        };
    }
};

// register with the global registry
MachineRegistry.register('lathe', LatheModule);
