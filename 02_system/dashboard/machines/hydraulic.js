// machines/hydraulic.js
// Hydraulic Unit machine type definition
// Handles rendering of hydraulic component cards for the supervisor dashboard

const HydraulicModule = {

    renderListItem(name, state, isSelected) {
        const status  = (state?.unit_status || 'NO DATA').toLowerCase().replace(' ', '');
        const cycle   = state?.cycle || 0;
        const total   = state?.total || 0;
        const pct     = total > 0 ? (cycle / total * 100) : 0;
        const barCol  = status === 'critical' ? 'var(--critical)' : status === 'warning' ? '#e08000' : 'var(--healthy)';
        const display = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const sel     = isSelected ? 'selected' : '';

        return `<div class="mc ${status} ${sel}" onclick="selectMachine('${name}','hydraulic')">
            <div class="mc-type">Hydraulic Unit</div>
            <div class="mc-name">${display}</div>
            <div class="mc-status ${status}">${state?.unit_status || 'NO DATA'}</div>
            <div class="mc-meta">${cycle} / ${total} cycles</div>
            <div class="mc-bar-bg"><div class="mc-bar" style="width:${pct.toFixed(1)}%;background:${barCol}"></div></div>
        </div>`;
    },

    renderHeatmapRow(name, state) {
        const label  = name.replace('hydraulic_', 'H');
        const status = (state?.unit_status || 'no-data').toLowerCase();
        const short  = status === 'no-data' ? '—' : status.slice(0, 3).toUpperCase();
        return `<div class="hm-row">
            <div class="hm-lathe-label">${label}</div>
            <div class="hm-cell ${status}" style="flex:4" title="${name}: ${status}">${short}</div>
        </div>`;
    },

    renderDetail(name, state) {
        if (!state) return;

        const status  = state.unit_status || 'NO DATA';
        const cycle   = state.cycle  || 0;
        const total   = state.total  || 0;
        const pct     = total > 0 ? (cycle / total * 100) : 0;
        const barCol  = status === 'CRITICAL' ? 'var(--critical)' : status === 'WARNING' ? '#e08000' : 'var(--healthy)';
        const display = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

        document.getElementById('dh-name').textContent             = display;
        document.getElementById('dh-sub').textContent              = 'UCI Hydraulic Systems — XGBoost Classifier';
        document.getElementById('detail-progress-txt').textContent = `Cycle ${cycle} / ${total} (${pct.toFixed(1)}%)`;
        document.getElementById('detail-progress-bar').style.cssText = `width:${pct.toFixed(1)}%;background:${barCol}`;

        const badge       = document.getElementById('dh-badge');
        badge.className   = `status-badge ${status.toLowerCase()}`;
        badge.textContent = status;

        document.getElementById('bearing-grid').style.display   = 'none';
        document.getElementById('hydraulic-grid').style.display = 'grid';

        // component cards
        const components = state.components || {};
        document.getElementById('hydraulic-grid').innerHTML = Object.entries(components).map(([cname, c]) => {
            const cStatus = (c.status || 'HEALTHY').toLowerCase();
            const conf    = c.confidence !== undefined ? `${c.confidence}% confidence` : '';
            return `<div class="hc ${cStatus}">
                <div class="hc-name">${cname.charAt(0).toUpperCase() + cname.slice(1)}</div>
                <div class="hc-condition ${cStatus}">${c.label || 'Unknown'}</div>
                ${conf ? `<div class="hc-confidence">${conf}</div>` : ''}
            </div>`;
        }).join('');

        // live sensor readings
        const sens   = state.sensors;
        const sensEl = document.getElementById('hyd-sensors');
        if (sens) {
            sensEl.style.display = 'grid';
            sensEl.innerHTML = [
                { key: 'pressure_bar',    label: 'Pressure',     unit: 'bar'  },
                { key: 'flow_lpm',        label: 'Flow Rate',    unit: 'l/min'},
                { key: 'temperature_c',   label: 'Temperature',  unit: '°C'   },
                { key: 'vibration_mms',   label: 'Vibration',    unit: 'mm/s' },
                { key: 'cooling_eff_pct', label: 'Cooling Eff.', unit: '%'    },
                { key: 'motor_power_w',   label: 'Motor Power',  unit: 'W'    },
            ].map(({ key, label, unit }) => `
                <div class="hyd-sensor">
                    <label>${label}</label>
                    <span class="val">${sens[key] !== undefined ? sens[key] : '—'}</span>
                    <span class="unit">${unit}</span>
                </div>`).join('');
        } else {
            sensEl.style.display = 'none';
        }
    },

    // hydraulic machines do not have history charts
    getHistoryDatasets() {
        return null;
    },

    getStatusSummary(name, state) {
        if (!state) return null;
        const components = state.components || {};
        const issues     = Object.entries(components)
            .filter(([, c]) => c.status !== 'HEALTHY')
            .map(([cname, c]) => `${cname}: ${c.label || c.status}`);
        return {
            name:    name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            type:    'Hydraulic Unit',
            status:  state.unit_status || 'NO DATA',
            issues,
            reading: `${state.cycle || 0} / ${state.total || 0} cycles`
        };
    }
};

MachineRegistry.register('hydraulic', HydraulicModule);
