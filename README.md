# Low-Cost Predictive Maintenance for SME Manufacturing

**Digital Systems Project — Charles Rodway (23048890), UWE Bristol (UFCFXK-30-3)**

An open-source, modular machine health monitoring system that uses machine learning to perform predictive maintenance and fault classification. Designed for Small and Medium Enterprises (SMEs) who cannot afford commercial predictive maintenance platforms.

The system deploys in Docker containers simulating Raspberry Pi edge devices, communicating via RabbitMQ, with a real-time web dashboard for monitoring machine health.

---

## What It Does

The system monitors two types of industrial equipment:

- **CNC Lathes (Bearing Monitoring)** — 3 simulated lathes, each with 4 bearings. Isolation Forest models detect anomalies from vibration features and raise alerts when a bearing shows sustained degradation (50 consecutive anomalies). Achieved false positive rates of 1.0–1.2% and early warning times of 19–82 hours before failure.

- **Hydraulic Units (Fault Classification)** — 3 simulated units. XGBoost classifiers predict the condition of 4 components (cooler, valve, pump, accumulator) from sensor readings. Achieved F1 scores above 0.97 across all components.

Everything runs in Docker with no cloud dependency — the system is designed to run on a local factory network using edge computing.

---

## Architecture

```
Sensor Data (NASA IMS / UCI Hydraulic datasets)
        ↓
Edge Device Containers  (raspi_lathe.py / raspi_hydraulic.py)
  — loads pre-trained ML model
  — extracts features and runs inference locally
  — publishes predictions to RabbitMQ
        ↓
RabbitMQ  (message broker, fanout exchanges per machine)
        ↓
FastAPI Backend  (backend.py)
  — aggregates predictions from all 6 machines
  — maintains system state and reading history
  — logs CRITICAL alerts to storage
  — exposes REST API on port 8000
        ↓
nginx Dashboard  (http://localhost:3000)
  — login with role-based access
  — supervisor view (full dashboard with history charts, heatmap, alerts)
  — maintenance view (status overview and alerts only)
```

---

## Project Structure

```
cnc-predictive-maintenance/
├── 01_model_development/
│   ├── bearing_isolation_forest.ipynb   # IMS bearing anomaly detection
│   ├── hydraulic_classifier.ipynb       # UCI hydraulic fault classification
│   └── results/                         # trained models + comparison plots
│
├── 02_system/
│   ├── docker-compose.yml               # all 9 services defined here
│   ├── config/
│   │   ├── lathes.json                  # per-lathe config (bearings, dataset, status)
│   │   ├── hydraulics.json              # per-unit config (model bundle, status)
│   │   └── alerts.json                  # written at runtime by backend
│   ├── models/
│   │   ├── lathe_1/                     # 4 Isolation Forest .pkl files per lathe
│   │   ├── lathe_2/
│   │   ├── lathe_3/
│   │   └── hydraulic/                   # 3 XGBoost bundles (1 per unit)
│   ├── src/
│   │   ├── backend/backend.py           # FastAPI + RabbitMQ consumer threads
│   │   └── raspi/
│   │       ├── raspi_lathe.py           # bearing edge device (monitoring + collecting modes)
│   │       └── raspi_hydraulic.py       # hydraulic edge device
│   ├── dashboard/
│   │   ├── login.html                   # role-based authentication
│   │   ├── dashboard.html               # supervisor view
│   │   ├── maintenance.html             # maintenance staff view
│   │   ├── dashboard.js                 # shared core logic + MachineRegistry
│   │   └── machines/
│   │       ├── lathe.js                 # bearing monitoring module
│   │       └── hydraulic.js             # hydraulic monitoring module
│   └── nginx/nginx.conf
│
├── bearing_data/                        # NASA IMS dataset (not in repo)
└── hydraulic_data/                      # UCI Hydraulic dataset (not in repo)
```

---

## Requirements

- Docker Desktop
- The datasets are not included in the repo due to size. Place them at:
  - `bearing_data/` — [NASA IMS Bearing Dataset](https://data.nasa.gov/dataset/ims-bearings) (1st_test, 2nd_test, 3rd_test folders)
  - `hydraulic_data/` — [UCI Hydraulic Systems Dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) (.txt sensor files + profile.txt)

---

## How to Run

```bash
cd 02_system
docker-compose up --build
```

Open `http://localhost:3000` in Chrome.

```bash
docker-compose down    # to stop
```

> Each run starts fresh — the system replays the datasets from the beginning so you will see the full degradation sequence play out.

---

## Logins

| Username | Password | Role |
|----------|----------|------|
| supervisor | admin123 | Full dashboard — history charts, fleet heatmap, alerts, machine registration |
| maintenance | maint123 | Maintenance panel — machine status and critical alerts only |

> Hardcoded for demonstration. A production deployment would use proper authentication.

---

## Ports

| Port | Service |
|------|---------|
| 3000 | Dashboard (nginx) |
| 8000 | Backend REST API (FastAPI) |
| 5672 | RabbitMQ broker |
| 15672 | RabbitMQ management UI (guest/guest) |

---

## Adding a New Machine

To add a new lathe or hydraulic unit of a supported type, no code changes are needed:

1. Train models and place `.pkl` files in `models/`
2. Add a config entry to `lathes.json` or `hydraulics.json`
3. Add a service block to `docker-compose.yml`
4. Restart the system

The backend discovers machines from config at startup and the dashboard discovers them from API responses.

---

## Hydraulic Degradation Sequences

Each hydraulic unit simulates a different fault progression:

| Unit | Sequence |
|------|----------|
| Hydraulic 1 | Valve: Optimal → Small lag → Severe lag → Near failure |
| Hydraulic 2 | Pump: No leakage → Weak leakage → Severe leakage |
| Hydraulic 3 | Multi-component: Cooler and accumulator degrade together |

---

## Datasets

- **NASA IMS Bearing Dataset** — University of Cincinnati. 3 test rigs, 4 bearings each, run to failure at 2000 RPM with vibration data at 20kHz.
- **UCI Hydraulic Systems Dataset** — 2205 operating cycles, 17 sensors, 4 labelled components (Helwig, Pignanelli and Schütze, 2015).
