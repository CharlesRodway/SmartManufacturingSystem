# 02_system — CNC Predictive Maintenance System

Production system built on Docker, RabbitMQ, FastAPI and Streamlit.

## Prerequisites

- Docker Desktop installed and running
- Pre-trained models from `01_model_development/results/`

## Setup

### 1. Copy pre-trained models

Models from `01_model_development` need to be organised into per-lathe folders:

```
02_system/models/
├── lathe_1/
│   ├── bearing1_model.pkl
│   ├── bearing2_model.pkl
│   ├── bearing3_model.pkl
│   └── bearing4_model.pkl
├── lathe_2/
│   └── ...
└── lathe_3/
    └── ...
```

Run this from the repo root:

```bash
# Windows PowerShell
mkdir 02_system\models\lathe_1
mkdir 02_system\models\lathe_2
mkdir 02_system\models\lathe_3

copy 01_model_development\results\lathe_1_bearing1_model.pkl 02_system\models\lathe_1\bearing1_model.pkl
copy 01_model_development\results\lathe_1_bearing2_model.pkl 02_system\models\lathe_1\bearing2_model.pkl
copy 01_model_development\results\lathe_1_bearing3_model.pkl 02_system\models\lathe_1\bearing3_model.pkl
copy 01_model_development\results\lathe_1_bearing4_model.pkl 02_system\models\lathe_1\bearing4_model.pkl

copy 01_model_development\results\lathe_2_bearing1_model.pkl 02_system\models\lathe_2\bearing1_model.pkl
copy 01_model_development\results\lathe_2_bearing2_model.pkl 02_system\models\lathe_2\bearing2_model.pkl
copy 01_model_development\results\lathe_2_bearing3_model.pkl 02_system\models\lathe_2\bearing3_model.pkl
copy 01_model_development\results\lathe_2_bearing4_model.pkl 02_system\models\lathe_2\bearing4_model.pkl

copy 01_model_development\results\lathe_3_bearing1_model.pkl 02_system\models\lathe_3\bearing1_model.pkl
copy 01_model_development\results\lathe_3_bearing2_model.pkl 02_system\models\lathe_3\bearing2_model.pkl
copy 01_model_development\results\lathe_3_bearing3_model.pkl 02_system\models\lathe_3\bearing3_model.pkl
copy 01_model_development\results\lathe_3_bearing4_model.pkl 02_system\models\lathe_3\bearing4_model.pkl
```

### 2. Build and run

```bash
cd 02_system
docker-compose up --build
```

### 3. Access the system

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| RabbitMQ Management | http://localhost:15672 (admin/password) |

## Architecture

```
[lathe_1_raspi] ──┐
[lathe_2_raspi] ──┼──► [RabbitMQ] ──► [Backend API] ──► [Dashboard]
[lathe_3_raspi] ──┘                        │
                                           └──► [Trainer] (watches for new lathes)
```

## Adding a new lathe

1. Open the dashboard at http://localhost:8501
2. Toggle "Register New Lathe" in the sidebar
3. Fill in the lathe details and click Register
4. Connect the Raspberry Pi to the network
5. The Pi will begin streaming data automatically
6. Once enough data is collected, click "Confirm machine was healthy"
7. The trainer service will auto-train models and switch to monitoring mode

## Stopping the system

```bash
docker-compose down
```
