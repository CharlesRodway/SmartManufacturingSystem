# Low-Cost Predictive Maintenance for SME Manufacturing

**Digital Systems Project — Charles Rodway (23048890), UWE Bristol (UFCFXK-30-3)**

An open-source, modular machine health monitoring system that uses machine learning to perform predictive maintenance and fault classification. Designed for Small and Medium Enterprises (SMEs) who cannot afford commercial predictive maintenance platforms.

The system deploys in Docker containers simulating Raspberry Pi edge devices, communicating via RabbitMQ, with a Grafana dashboard for real-time monitoring of machine health.

---

## What It Does

The system monitors CNC lathes using bearing vibration data:

- **CNC Lathes (Bearing Monitoring)** — 3 simulated lathes, each with 4 bearings. Isolation Forest models detect anomalies from vibration features and raise alerts when a bearing shows sustained degradation (50 consecutive anomalies). Achieved false positive rates of 1.0–1.2% and early warning times of 19–82 hours before failure.

Everything runs in Docker with no cloud dependency — the system is designed to run on a local factory network using edge computing.

---

## Architecture

```
Sensor Data (NASA IMS bearing dataset)
        ↓
Edge Device Containers  (raspi_lathe.py)
  — loads pre-trained ML model
  — extracts features and runs inference locally
  — publishes predictions to RabbitMQ
        ↓
RabbitMQ  (message broker, fanout exchanges per machine)
        ↓
FastAPI Backend  (backend.py)
  — aggregates predictions from all 3 machines
  — maintains system state and reading history
  — logs CRITICAL alerts to storage
  — exposes REST API on port 8000
  — serves SimpleJSON endpoints for Grafana
        ↓
Grafana  (http://localhost:3000)
  — connects to backend via SimpleJSON datasource
  — build dashboards with time series, status panels, and alerts
  — login: admin / admin
```

---

## Project Structure

```
cnc-predictive-maintenance/
├── 01_model_development/
│   ├── bearing_isolation_forest.ipynb   # IMS bearing anomaly detection
│   └── results/                         # trained models + comparison plots
│
├── 02_system/
│   ├── docker-compose.yml               # all 6 services defined here
│   ├── config/
│   │   ├── lathes.json                  # per-lathe config (bearings, dataset, status)
│   │   └── alerts.json                  # written at runtime by backend
│   ├── models/
│   │   ├── lathe_1/                     # 4 Isolation Forest .pkl files per lathe
│   │   ├── lathe_2/
│   │   └── lathe_3/
│   ├── src/
│   │   ├── backend/backend.py           # FastAPI + RabbitMQ consumer threads
│   │   └── raspi/
│   │       └── raspi_lathe.py           # bearing edge device (monitoring + collecting modes)
│   └── grafana/
│       └── provisioning/
│           └── datasources/
│               └── datasource.yml       # auto-provisions SimpleJSON datasource
│
└── bearing_data/                        # NASA IMS dataset (not in repo)
```

---

## Requirements

- Docker Desktop
- The dataset is not included in the repo due to size. Place it at:
  - `bearing_data/` — [NASA IMS Bearing Dataset](https://data.nasa.gov/dataset/ims-bearings) (1st_test, 2nd_test, 3rd_test folders)

---

## How to Run

```bash
cd 02_system
docker-compose up --build
```

Open `http://localhost:3000` in your browser — this is Grafana. Log in with `admin / admin`.

The SimpleJSON datasource (`CNC Backend`) is provisioned automatically. Create a new dashboard, add a time series panel, and select metrics from the datasource (e.g. `lathe_1.bearing1.score`).

```bash
docker-compose down    # to stop
```

> Each run starts fresh — the system replays the dataset from the beginning so you will see the full degradation sequence play out.

---

## Ports

| Port | Service |
|------|---------|
| 3000 | Grafana (admin / admin) |
| 8000 | Backend REST API (FastAPI) |
| 5672 | RabbitMQ broker |
| 15672 | RabbitMQ management UI (guest/guest) |

---

## Adding a New Machine

To add a new lathe, no code changes are needed:

1. Train models and place `.pkl` files in `models/`
2. Add a config entry to `lathes.json`
3. Add a service block to `docker-compose.yml`
4. Restart the system

The backend discovers machines from config at startup. The `/search` endpoint will expose new metric names to Grafana automatically once messages arrive.

---

## Dataset

- **NASA IMS Bearing Dataset** — University of Cincinnati. 3 test rigs, 4 bearings each, run to failure at 2000 RPM with vibration data at 20kHz.
