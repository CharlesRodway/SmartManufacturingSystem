# CNC Predictive Maintenance System
**Digital Systems Project — Charles Rodway, UWE Bristol (UFCFXK-30-3)**

A simulated IoT predictive maintenance system for CNC machines. The system uses machine learning to detect bearing anomalies and classify hydraulic faults in real time, with data flowing from simulated edge devices through a message broker to a web dashboard.

---

## What it does

The system monitors two types of machine:

- **CNC Lathes** — 3 simulated lathes, each with 4 bearings. Isolation Forest models detect anomalies from vibration features and raise alerts when a bearing shows sustained degradation.
- **Hydraulic Units** — 3 simulated units. XGBoost classifiers predict the condition of 4 components (cooler, valve, pump, accumulator) from sensor readings.

Everything runs in Docker. There is no cloud dependency — the system is designed to run on a local factory network.

---

## Architecture

```
NASA IMS / UCI sensor data
        ↓
Raspberry Pi containers  (raspi_lathe.py / raspi_hydraulic.py)
  — loads ML model locally
  — runs inference on each reading
  — publishes result to RabbitMQ
        ↓
RabbitMQ  (message broker)
        ↓
FastAPI backend  (backend.py)
  — aggregates results
  — holds state in memory
  — logs critical alerts
  — exposes REST API on port 8000
        ↓
nginx dashboard  (http://localhost:3000)
  — login page
  — supervisor view (full dashboard)
  — maintenance view (alerts + status only)
```

---

## Project structure

```
cnc-predictive-maintenance/
├── 01_model_development/
│   ├── bearing_isolation_forest.ipynb   # IMS bearing anomaly detection
│   ├── hydraulic_classifier.ipynb       # UCI hydraulic fault classification
│   └── results/                         # trained models + plots
│
├── 02_system/
│   ├── docker-compose.yml
│   ├── config/
│   │   ├── lathes.json
│   │   ├── hydraulics.json
│   │   └── alerts.json                  # written at runtime
│   ├── models/                          # model bundles loaded by edge containers
│   ├── src/
│   │   ├── backend/backend.py
│   │   └── raspi/raspi_lathe.py
│   │         raspi_hydraulic.py
│   ├── dashboard/                       # served by nginx
│   │   ├── login.html
│   │   ├── dashboard.html               # supervisor view
│   │   ├── maintenance.html             # maintenance view
│   │   ├── dashboard.js                 # shared core logic
│   │   └── machines/
│   │       ├── lathe.js                 # bearing card module
│   │       └── hydraulic.js             # hydraulic card module
│   └── nginx/nginx.conf
│
├── bearing_data/                        # NASA IMS dataset (not in repo)
└── hydraulic_data/                      # UCI Hydraulic dataset (not in repo)
```

---

## Requirements

- Docker Desktop
- The datasets are not included in the repo due to size. Place them at:
  - `bearing_data/` — NASA IMS Bearing Dataset (1st_test, 2nd_test, 3rd_test folders)
  - `hydraulic_data/` — UCI Hydraulic Systems Dataset (.txt sensor files + profile.txt)

---

## How to run

```bash
cd 02_system
docker-compose up --build
```

Then open `http://localhost:3000` in Chrome.

To stop:
```bash
docker-compose down
```

> **Note:** Each run starts fresh. The system replays the datasets from the beginning each time so you will always see the full degradation sequence play out.

---

## Logins

| Username | Password | Role |
|----------|----------|------|
| supervisor | admin123 | Full dashboard — all machines, history charts, alerts |
| maintenance | maint123 | Maintenance panel — alerts and machine status only |

> These are hardcoded for demonstration purposes. A production deployment would use a proper authentication service.

---

## Ports

| Port | Service |
|------|---------|
| 3000 | Dashboard (nginx) |
| 8000 | Backend API (FastAPI) |
| 5672 | RabbitMQ broker |
| 15672 | RabbitMQ management UI (guest/guest) |

---

## Hydraulic unit degradation sequences

Each hydraulic unit is configured to simulate a different fault progression during the demo:

| Unit | Story |
|------|-------|
| Hydraulic 1 | Valve degrades: Optimal → Small lag → Severe lag → Near failure |
| Hydraulic 2 | Pump leakage: No leakage → Weak leakage → Severe leakage |
| Hydraulic 3 | Multi-component: Cooler and accumulator degrade together |

---

## Branches

| Branch | Description |
|--------|-------------|
| main | Stable working system |
| modular-dashboard | Modular dashboard with nginx, login system, and custom hydraulic sequences |

---

## Datasets used

- **NASA IMS Bearing Dataset** — University of Cincinnati, recorded until bearing failure at 2000 RPM
- **UCI Hydraulic Systems Dataset** — 2205 operating cycles across 17 sensors, 4 labelled components

---

*BSc Computer Science, Year 3 — UWE Bristol, 2025/26*
