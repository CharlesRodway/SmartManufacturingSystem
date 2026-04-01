# Raspberry Pi Bearing Monitor - Production Version
# Digital Systems Project - Charles Rodway
#
# loads isolation forest models per bearing and streams vibration data, runs inference on them and published results to rabbitmq

import os
import sys
import time
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import pika
from scipy import stats
from pathlib import Path
from datetime import datetime


# settings - mostly from environment variables so they can be set per-container

LATHE_NAME = os.environ.get("LATHE_NAME", "lathe_1")
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "password")
STREAM_DELAY = float(os.environ.get("STREAM_DELAY", "0.5"))
STREAM_START_PERCENT = float(os.environ.get("STREAM_START_PERCENT", "0"))

CONFIG_PATH = Path("/app/config/lathes.json")
MODELS_DIR = Path("/app/models")
BEARING_DATA_DIR = Path("/app/bearing_data")
COLLECTED_DATA_DIR = Path("/app/data")

TRAIN_SPLIT = 0.2
ANOMALY_STREAK_THRESHOLD = 50

# column names depend on how many channels the dataset has
COLS_8 = [
    'bearing1_ch1', 'bearing1_ch2',
    'bearing2_ch1', 'bearing2_ch2',
    'bearing3_ch1', 'bearing3_ch2',
    'bearing4_ch1', 'bearing4_ch2'
]

COLS_4 = [
    'bearing1_ch1',
    'bearing2_ch1',
    'bearing3_ch1',
    'bearing4_ch1'
]


# feature extraction - same approach as used during training

def calculate_features(signal):
    signal = np.array(signal, dtype=np.float64)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak_to_peak = max_val - min_val
    crest_factor = max_val / rms if rms != 0 else 0
    kurtosis = stats.kurtosis(signal, fisher=True)
    skewness = stats.skew(signal)
    mean_abs = np.mean(np.abs(signal))
    shape_factor = rms / mean_abs if mean_abs != 0 else 0
    impulse_factor = max_val / mean_abs if mean_abs != 0 else 0
    return {
        'mean': mean_val, 'std': std_val, 'max': max_val, 'min': min_val,
        'rms': rms, 'peak_to_peak': peak_to_peak, 'crest_factor': crest_factor,
        'kurtosis': kurtosis, 'skewness': skewness,
        'shape_factor': shape_factor, 'impulse_factor': impulse_factor
    }


def load_file(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None)
    n_cols = df.shape[1]
    if n_cols == 8:
        df.columns = COLS_8
    elif n_cols == 4:
        df.columns = COLS_4
    else:
        df.columns = [f'ch{i}' for i in range(n_cols)]
    return df


def extract_bearing_features(filepath, channels):
    df = load_file(filepath)
    all_channel_features = []
    for ch in channels:
        if ch in df.columns:
            feats = calculate_features(df[ch].values)
            all_channel_features.append(feats)
    if not all_channel_features:
        return None
    # average across channels if there are multiple sensors on one bearing
    averaged = {}
    for key in all_channel_features[0].keys():
        averaged[key] = np.mean([f[key] for f in all_channel_features])
    return averaged


# load the pre-trained isolation forest models for this lathe

def load_models(lathe_name, bearings):
    models = {}
    for bearing_name in bearings.keys():
        model_path = MODELS_DIR / lathe_name / f"{bearing_name}_model.pkl"
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            print("Make sure models are copied to 02_system/models/")
            sys.exit(1)
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        models[bearing_name] = bundle
        print(f"  Loaded: {model_path.name}")
    return models


# rabbitmq connection with retry logic

def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )
    for attempt in range(10):
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            # one exchange per lathe, fanout so multiple consumers can subscribe
            channel.exchange_declare(
                exchange=LATHE_NAME,
                exchange_type='fanout',
                durable=True
            )
            print(f"Connected to RabbitMQ at {RABBITMQ_HOST}")
            return connection, channel
        except Exception as e:
            print(f"RabbitMQ not ready, retrying ({attempt+1}/10)... {e}")
            time.sleep(5)
    print("ERROR: Could not connect to RabbitMQ after 10 attempts")
    sys.exit(1)


def publish_message(channel, message):
    channel.basic_publish(
        exchange=LATHE_NAME,
        routing_key='',
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type='application/json'
        )
    )


# determine bearing health status based on anomaly score and streak

def get_health_status(score, streak, latched):
    if latched:
        return "CRITICAL"
    if streak >= ANOMALY_STREAK_THRESHOLD:
        return "CRITICAL"
    elif score < 0:
        return "WARNING"
    return "HEALTHY"


def get_machine_status(bearing_statuses):
    # overall machine status is the worst bearing
    if "CRITICAL" in bearing_statuses.values():
        return "CRITICAL"
    elif "WARNING" in bearing_statuses.values():
        return "WARNING"
    return "HEALTHY"


def update_collection_progress(hours_completed):
    # update lathes.json with how many hours of data have been collected
    # the trainer watches this to know when to start training
    try:
        with open(CONFIG_PATH) as f:
            all_config = json.load(f)
        all_config[LATHE_NAME]["collection_hours_completed"] = round(hours_completed, 2)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(all_config, f, indent=2)
    except Exception as e:
        print(f"Warning: could not update collection progress: {e}")


def run_collecting_mode(config, data_dir):
    # in collecting mode we just copy raw data files to the collection directory
    # no inference, no models needed - just accumulating healthy baseline data
    # in real life this would be writing live sensor readings to files instead
    print(f"\nMode: COLLECTING - saving data for future training")

    output_dir = COLLECTED_DATA_DIR / LATHE_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to: {output_dir}")

    hours_required = config.get("collection_hours_required", 336)

    # use first 80% of dataset as the healthy collection data
    all_files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    collection_files = all_files[:int(len(all_files) * TRAIN_SPLIT * 4)]

    # each file represents 20 minutes of real sensor data
    minutes_per_file = 20
    total_hours = (len(collection_files) * minutes_per_file) / 60

    print(f"Simulating {len(collection_files)} files = {total_hours:.1f} hours of data collection")
    print(f"Required: {hours_required} hours\n")

    for i, filepath in enumerate(collection_files):
        dest = output_dir / filepath.name
        shutil.copy2(filepath, dest)

        hours_done = ((i + 1) * minutes_per_file) / 60
        progress = min((hours_done / hours_required) * 100, 100)

        # update progress in lathes.json every 10 files so dashboard stays current
        if (i + 1) % 10 == 0 or (i + 1) == len(collection_files):
            update_collection_progress(hours_done)
            print(f"  Collected {i+1}/{len(collection_files)} files "
                  f"({hours_done:.1f}h / {hours_required}h) [{progress:.1f}%]")

        time.sleep(STREAM_DELAY)

    print(f"\nData collection complete. {total_hours:.1f} hours collected.")
    print("Waiting for engineer to confirm machine was healthy...")

    # keep the container alive after collection finishes
    while True:
        time.sleep(60)


def run_monitoring_mode(config, data_dir):
    # monitoring mode - load models, run inference, publish to rabbitmq
    print(f"\nMode: MONITORING - running live inference")

    bearings = config["bearings"]

    print("\nLoading models...")
    models = load_models(LATHE_NAME, bearings)

    print("\nConnecting to RabbitMQ...")
    connection, channel = connect_rabbitmq()

    # skip training portion, then optionally skip ahead further for demo purposes
    all_files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    split_idx = int(len(all_files) * TRAIN_SPLIT)
    test_files = all_files[split_idx:]

    # STREAM_START_PERCENT lets you jump into the dataset at a specific point
    # e.g. set to 75 to start at 75% through the test data, useful for demos
    if STREAM_START_PERCENT > 0:
        start_idx = int(len(test_files) * (STREAM_START_PERCENT / 100))
        test_files = test_files[start_idx:]
        print(f"Demo mode: starting at {STREAM_START_PERCENT}% through test data "
              f"(skipping {start_idx} files)")

    total = len(test_files)

    print(f"\nSkipping {split_idx} training files")
    print(f"Streaming {total} unseen files\n")

    # per-bearing state tracking
    bearing_streaks = {b: 0 for b in bearings}
    bearing_latched = {b: False for b in bearings}
    bearing_alerts = {b: False for b in bearings}
    bearing_alert_readings = {b: None for b in bearings}
    first_anomaly_readings = {b: None for b in bearings}

    try:
        for i, filepath in enumerate(test_files):
            reading_num = i + 1
            bearing_results = {}
            bearing_statuses = {}

            for bearing_name, channels in bearings.items():
                features = extract_bearing_features(filepath, channels)
                if features is None:
                    continue

                bundle = models[bearing_name]
                model = bundle['model']
                feature_names = bundle['feature_names']

                X = np.array([[features[n] for n in feature_names]])
                pred = model.predict(X)[0]
                score = model.decision_function(X)[0]
                is_anomaly = (pred == -1)

                # update streak - once latched the streak stays frozen
                if not bearing_latched[bearing_name]:
                    if is_anomaly:
                        bearing_streaks[bearing_name] += 1
                        if first_anomaly_readings[bearing_name] is None:
                            first_anomaly_readings[bearing_name] = reading_num
                    else:
                        bearing_streaks[bearing_name] = 0

                streak = bearing_streaks[bearing_name]

                # latch to critical once streak threshold is hit
                if streak >= ANOMALY_STREAK_THRESHOLD and not bearing_alerts[bearing_name]:
                    bearing_alerts[bearing_name] = True
                    bearing_latched[bearing_name] = True
                    bearing_alert_readings[bearing_name] = reading_num
                    print(f"ALERT: {bearing_name.upper()} LATCHED CRITICAL at reading {reading_num}")

                status = get_health_status(score, streak, bearing_latched[bearing_name])

                bearing_results[bearing_name] = {
                    'status': status,
                    'score': round(score, 4),
                    'kurtosis': round(features.get('kurtosis', 0), 3),
                    'rms': round(features.get('rms', 0), 4),
                    'streak': streak,
                    'latched': bearing_latched[bearing_name],
                    'is_anomaly': bool(is_anomaly)
                }
                bearing_statuses[bearing_name] = status

            machine_status = get_machine_status(bearing_statuses)

            message = {
                'lathe': LATHE_NAME,
                'reading': reading_num,
                'total': total,
                'timestamp': datetime.now().isoformat(),
                'filename': filepath.name,
                'machine_status': machine_status,
                'bearings': bearing_results
            }

            try:
                publish_message(channel, message)
            except Exception as e:
                print(f"Publish error: {e}, reconnecting...")
                connection, channel = connect_rabbitmq()
                publish_message(channel, message)

            progress = (reading_num / total) * 100
            print(f"[{reading_num}/{total}] ({progress:.1f}%) "
                  f"Machine: {machine_status} | "
                  f"Bearings: { {b: r['status'] for b, r in bearing_results.items()} }")

            time.sleep(STREAM_DELAY)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        connection.close()
        print("RabbitMQ connection closed")


def main():
    print(f"\n{'='*60}")
    print(f"CNC PREDICTIVE MAINTENANCE - RASPI SERVICE")
    print(f"{'='*60}")
    print(f"Lathe:    {LATHE_NAME.upper()}")
    print(f"RabbitMQ: {RABBITMQ_HOST}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # load this lathe's config from the shared config file
    with open(CONFIG_PATH) as f:
        all_config = json.load(f)

    if LATHE_NAME not in all_config:
        print(f"ERROR: {LATHE_NAME} not found in lathes.json")
        sys.exit(1)

    config = all_config[LATHE_NAME]
    status = config.get("status", "monitoring")
    dataset_path = config["dataset_path"]
    data_dir = BEARING_DATA_DIR / dataset_path

    print(f"Status:   {status}")
    print(f"Dataset:  {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        sys.exit(1)

    # route to correct mode based on lathe status
    if status == "collecting":
        run_collecting_mode(config, data_dir)
    else:
        run_monitoring_mode(config, data_dir)


if __name__ == "__main__":
    main()