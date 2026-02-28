# 🔧 CNC Predictive Maintenance
### Bearing Fault Detection using Machine Learning & IoT

> **Early bearing failure detection ~4 days in advance** using Isolation Forest anomaly detection, deployed across Raspberry Pi edge devices with RabbitMQ messaging, Docker containerisation, and a real-time Streamlit dashboard.

---

## 📌 Overview

Traditional CNC machine maintenance is either **reactive** (fix it when it breaks) or **scheduled** (replace parts on a timer). Both are costly — reactive maintenance causes unplanned downtime, scheduled maintenance wastes parts that are still healthy.

This project implements a **smart predictive maintenance system** that continuously monitors CNC machine bearing health using vibration data, applies unsupervised machine learning at the edge to detect anomalies before failure occurs, and routes alerts through a message broker to a centralised monitoring dashboard.

The system is designed to scale across multiple machines — each CNC lathe has a dedicated Raspberry Pi running local inference, publishing results to a shared RabbitMQ broker, with all data surfaced through a single Streamlit dashboard for operators and technicians.

---

## 🤖 Machine Learning

The core of this project is an **Isolation Forest** model for unsupervised anomaly detection on bearing vibration signals.

### Why Isolation Forest?
- No labelled failure data required — the model learns what *normal* looks like and flags deviations
- Effective at detecting subtle changes in vibration patterns that precede failure
- Computationally lightweight — well suited for edge deployment on a Raspberry Pi

### Feature Engineering
Each raw vibration file is processed into **88 statistical features** (11 features × 8 channels across 4 bearings):

| Feature | Description |
|---|---|
| RMS | Overall vibration energy |
| Kurtosis | Key indicator of bearing faults — spikes sharply before failure |
| Crest Factor | Ratio of peak to RMS — sensitive to impulsive events |
| Peak-to-Peak | Total signal amplitude range |
| Shape Factor | RMS / mean absolute value |
| Impulse Factor | Peak / mean absolute value |
| Skewness | Asymmetry of the signal distribution |
| Mean, Std, Max, Min | Basic statistical descriptors |

### Training Approach
- Trained on the **first 20% of the dataset** (healthy operation period only)
- The model learns a baseline of normal bearing behaviour
- The remaining **80% is left entirely unseen** — used to simulate live inference on the Raspberry Pi
- `contamination=0.01` reflects the expectation that training data contains very few anomalies

### Dataset
The **NASA IMS Bearing Dataset** consists of three run-to-failure tests with four bearings each, run continuously under constant load. Each file contains 8 channels of vibration data sampled at 20,480 Hz, recorded approximately every 10 minutes. In the 1st test, Bearing 3 develops an inner race defect and Bearing 4 a rolling element defect.

> Dataset available from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 📊 Results

- ✅ Alert triggered via **30 consecutive anomaly streak threshold**
- ✅ System provided **~4 days advance warning** before confirmed bearing failure
- ✅ Kurtosis on Bearing 3 spikes sharply in the final stage — clearly visible in plots
- ✅ Anomaly scores trend downward (more anomalous) as bearing degrades over time

> See [`01_model_development/results/anomaly_analysis.png`](./01_model_development/results/anomaly_analysis.png) for the full 4-panel results plot.

The visualisation script generates four plots:
1. **Anomaly rate over time** — rises sharply near failure
2. **Kurtosis** — Bearings 3 & 4 spike vs. healthy baseline (~3)
3. **Anomaly scores** — normal (blue) vs. anomalous (red) readings over test progress
4. **RMS energy levels** — vibration energy increase in failing bearings vs. healthy

---

## 🏗️ System Architecture

![CNC Health Monitoring Architecture](./docs/SystemArchitecture.png)

Each Raspberry Pi sits at a CNC lathe, runs local feature extraction and Isolation Forest inference, and publishes anomaly alerts and feature data to a RabbitMQ message broker. The broker feeds a backend service running inside Docker, which persists results and drives a Streamlit dashboard for real-time operator monitoring and a technician interface for threshold configuration and alert history.

### Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Edge Device | Raspberry Pi | On-site data acquisition & ML inference |
| ML Model | Isolation Forest (scikit-learn) | Unsupervised anomaly detection |
| Message Broker | RabbitMQ | Async alert routing from Pi to server |
| Containerisation | Docker / Docker Compose | Service orchestration & portability |
| Dashboard | Streamlit | Real-time monitoring & visualisation UI |
| Language | Python | End-to-end implementation |

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- NASA IMS Bearing Dataset placed at `bearing_data/`

### Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Train the model
```bash
python 01_model_development/train_isolation_forest.py
```

### Step 2 — Start the Docker stack
```bash
docker-compose up --build
```
Dashboard available at `http://localhost:8501`.

### Step 3 — Run the Raspberry Pi simulation
```bash
python 02_system/src/raspi.py
```

### Step 4 — Generate results plots
```bash
python 01_model_development/visualisation.py
```

---

## 📁 Project Structure

```
cnc-predictive-maintenance/
│
├── 01_model_development/
│   ├── train_isolation_forest.py
│   ├── raspi.py
│   ├── visualisation.py
│   └── results/
│       ├── training_results.csv
│       └── anomaly_analysis.png
│
├── 02_system/
│   ├── src/
│   │   └── raspi.py
│   ├── dashboard/
│   │   └── streamlit_app.py
│   ├── models/
│   │   └── isolation_forest_model.pkl
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── bearing_data/
│   ├── 1st_test/
│   ├── 2nd_test/
│   └── 3rd_test/
│
├── docs/
│   └── SystemArchitecture.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The `bearing_data/` folder is not included in this repository due to file size (~6GB). Download the NASA IMS Bearing Dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place the test folders as shown above.

---

## 💰 Cost vs. Commercial Solutions

| Solution | Approximate Cost |
|---|---|
| Commercial PdM System | £5,000 – £20,000+ |
| This System (Raspberry Pi + sensors) | ~£300 – £600 |

**16–28× cost savings** while maintaining comparable detection capability.

---

## 🎓 Academic Context

Developed as a **Final Year Dissertation** for BSc Computer Science at the **University of the West of England (UWE Bristol)**, specialising in AI, Machine Learning, and IoT/Smart Devices.

---

## 📚 References

- NASA IMS Bearing Dataset — University of Cincinnati IMS Center
- Liu, F.T., Ting, K.M., Zhou, Z-H. (2008). *Isolation Forest.* IEEE ICDM.
