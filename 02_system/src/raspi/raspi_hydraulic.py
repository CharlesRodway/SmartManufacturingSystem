# Hydraulic Unit Edge Simulator
# Digital Systems Project - Charles Rodway
#
# Simulates a Raspberry Pi edge device for a hydraulic monitoring unit.
# Replays the UCI Hydraulic Systems dataset cycle by cycle, runs inference
# using the pre-trained XGBoost bundle, and publishes results to RabbitMQ.
#
# NOTE: standalone edge simulation script, superseded by the Docker-based
# implementation in 02_system/. Retained for reference and isolated testing.
#
# Environment variables:
#   UNIT_NAME        - e.g. hydraulic_1
#   RABBITMQ_HOST    - hostname of RabbitMQ broker
#   RABBITMQ_USER    - RabbitMQ username
#   RABBITMQ_PASS    - RabbitMQ password
#   STREAM_DELAY     - seconds between cycles (default 0.5)

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pika

# ── Config ────────────────────────────────────────────────────────────────────

UNIT_NAME     = os.environ.get("UNIT_NAME", "hydraulic_1")
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "password")
STREAM_DELAY  = float(os.environ.get("STREAM_DELAY", "0.5"))

CONFIG_PATH  = Path("/app/config/hydraulics.json")
MODELS_DIR   = Path("/app/models/hydraulic")
DATA_DIR     = Path("/app/hydraulic_data")

# UCI sensor definitions — must match hydraulic_classifier.ipynb
SENSORS = [
    ('PS1',  100, 6000), ('PS2',  100, 6000), ('PS3',  100, 6000),
    ('PS4',  100, 6000), ('PS5',  100, 6000), ('PS6',  100, 6000),
    ('EPS1', 100, 6000), ('FS1',   10,  600), ('FS2',   10,  600),
    ('TS1',    1,   60), ('TS2',    1,   60), ('TS3',    1,   60),
    ('TS4',    1,   60), ('VS1',    1,   60), ('CE',     1,   60),
    ('CP',     1,   60), ('SE',     1,   60),
]
COMPONENTS = ['cooler', 'valve', 'pump', 'accumulator']


# ── Load config ───────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    if UNIT_NAME not in config:
        raise ValueError(f"Unit '{UNIT_NAME}' not found in hydraulics.json")
    return config[UNIT_NAME]


# ── Load model bundle ─────────────────────────────────────────────────────────

def load_bundle(bundle_name):
    path = MODELS_DIR / bundle_name
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"[{UNIT_NAME}] Loaded model bundle: {bundle_name}")
    return bundle


# ── Load sensor data ──────────────────────────────────────────────────────────

def load_sensor_data():
    print(f"[{UNIT_NAME}] Loading sensor files...")
    raw = {}
    for name, hz, pts in SENSORS:
        df = pd.read_csv(DATA_DIR / f'{name}.txt', sep='\t', header=None)
        assert df.shape == (2205, pts), f"{name}: unexpected shape {df.shape}"
        raw[name] = df
    profile = pd.read_csv(DATA_DIR / 'profile.txt', sep='\t', header=None,
                          names=['cooler', 'valve', 'pump', 'accumulator', 'stable'])
    print(f"[{UNIT_NAME}] Loaded {len(SENSORS)} sensors, 2205 cycles")
    return raw, profile


# ── Feature extraction — must match hydraulic_classifier.ipynb ────────────────

def extract_features(raw_sensors, cycle_idx):
    row = {}
    for name, hz, pts in SENSORS:
        vals = raw_sensors[name].iloc[cycle_idx].values.astype(np.float64)
        row[f'{name}_mean']  = float(np.mean(vals))
        row[f'{name}_std']   = float(np.std(vals))
        row[f'{name}_max']   = float(np.max(vals))
        row[f'{name}_min']   = float(np.min(vals))
        row[f'{name}_range'] = float(np.max(vals) - np.min(vals))
    return row


# ── Run inference ─────────────────────────────────────────────────────────────

def run_inference(bundle, features):
    scaler        = bundle['scaler']
    models        = bundle['component_models']
    encoders      = bundle['encoders']
    labels_map    = bundle['component_labels']
    feature_names = bundle['feature_names']

    X = np.array([[features[f] for f in feature_names]])
    X_scaled = scaler.transform(X)

    results = {}
    for component in COMPONENTS:
        model      = models[component]
        le         = encoders[component]
        local_pred = model.predict(X_scaled)[0]
        global_idx = model._local_to_global[local_pred]
        raw_val    = int(le.classes_[global_idx])
        label      = labels_map[component].get(raw_val, str(raw_val))
        results[component] = {
            "raw_value": raw_val,
            "label":     label,
            "status":    classify_severity(component, raw_val)
        }
    return results


def classify_severity(component, raw_value):
    """Map raw condition value to a severity status."""
    thresholds = {
        'cooler':      {3: 'CRITICAL', 20: 'WARNING', 100: 'HEALTHY'},
        'valve':       {73: 'CRITICAL', 80: 'WARNING', 90: 'WARNING', 100: 'HEALTHY'},
        'pump':        {0: 'HEALTHY', 1: 'WARNING', 2: 'CRITICAL'},
        'accumulator': {90: 'CRITICAL', 100: 'WARNING', 115: 'WARNING', 130: 'HEALTHY'},
    }
    return thresholds.get(component, {}).get(raw_value, 'HEALTHY')


def overall_status(components):
    statuses = [c['status'] for c in components.values()]
    if 'CRITICAL' in statuses:
        return 'CRITICAL'
    if 'WARNING' in statuses:
        return 'WARNING'
    return 'HEALTHY'


# ── RabbitMQ ──────────────────────────────────────────────────────────────────

def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters  = pika.ConnectionParameters(
        host=RABBITMQ_HOST, credentials=credentials,
        heartbeat=60, blocked_connection_timeout=30
    )
    for attempt in range(10):
        try:
            conn = pika.BlockingConnection(parameters)
            print(f"[{UNIT_NAME}] Connected to RabbitMQ")
            return conn
        except Exception as e:
            print(f"[{UNIT_NAME}] RabbitMQ not ready ({attempt+1}/10)... {e}")
            time.sleep(5)
    raise RuntimeError("Could not connect to RabbitMQ")


def publish(channel, message):
    channel.basic_publish(
        exchange=UNIT_NAME,
        routing_key='',
        body=json.dumps(message),
        properties=pika.BasicProperties(delivery_mode=2)
    )


# ── Main stream loop ──────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*50}")
    print(f"HYDRAULIC UNIT SIMULATOR — {UNIT_NAME}")
    print(f"{'='*50}")

    config     = load_config()
    bundle     = load_bundle(config['model_bundle'])
    raw, profile = load_sensor_data()

    # filter to stable cycles only — matches training
    stable_mask = profile['stable'] == 0
    stable_indices = profile.index[stable_mask].tolist()
    total = len(stable_indices)
    print(f"[{UNIT_NAME}] {total} stable cycles to stream")

    connection = connect_rabbitmq()
    channel    = connection.channel()
    channel.exchange_declare(exchange=UNIT_NAME, exchange_type='fanout', durable=True)

    print(f"[{UNIT_NAME}] Starting stream (delay={STREAM_DELAY}s per cycle)...")

    for cycle_num, cycle_idx in enumerate(stable_indices, start=1):
        try:
            features   = extract_features(raw, cycle_idx)
            components = run_inference(bundle, features)
            status     = overall_status(components)

            message = {
                "unit":        UNIT_NAME,
                "cycle":       cycle_num,
                "total":       total,
                "timestamp":   datetime.now().isoformat(),
                "unit_status": status,
                "components":  components
            }

            publish(channel, message)

            if cycle_num % 50 == 0 or status in ('WARNING', 'CRITICAL'):
                comp_summary = ' | '.join(
                    f"{c}: {v['label']}" for c, v in components.items()
                )
                print(f"[{UNIT_NAME}] Cycle {cycle_num}/{total} [{status}] {comp_summary}")

            time.sleep(STREAM_DELAY)

        except pika.exceptions.AMQPConnectionError:
            print(f"[{UNIT_NAME}] Connection lost, reconnecting...")
            connection = connect_rabbitmq()
            channel    = connection.channel()
            channel.exchange_declare(exchange=UNIT_NAME, exchange_type='fanout', durable=True)

    print(f"[{UNIT_NAME}] Stream complete — {total} cycles published")


if __name__ == "__main__":
    run()
