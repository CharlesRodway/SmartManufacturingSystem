# Raspberry Pi Bearing Monitor - Per-Bearing Live Inference
# Digital Systems Project - Charles Rodway
#
# Simulates a Raspberry Pi monitoring a CNC lathe.
# Loads one Isolation Forest model per bearing and runs predictions independently.
# Calculates overall machine health from individual bearing scores.
# Publishes per-bearing alerts via RabbitMQ (when integrated).
#
# Usage: python raspi.py --lathe lathe_1
#        python raspi.py --lathe lathe_2
#        python raspi.py --lathe lathe_3

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime


# ============ SETTINGS ============

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

LATHE_CONFIG = {
    "lathe_1": {
        "data_dir": BASE_DIR / "bearing_data" / "1st_test",
        "bearings": {
            "bearing1": ["bearing1_ch1", "bearing1_ch2"],
            "bearing2": ["bearing2_ch1", "bearing2_ch2"],
            "bearing3": ["bearing3_ch1", "bearing3_ch2"],
            "bearing4": ["bearing4_ch1", "bearing4_ch2"],
        },
        "known_failures": {
            "bearing3": "Inner Race Defect",
            "bearing4": "Rolling Element Defect",
        }
    },
    "lathe_2": {
        "data_dir": BASE_DIR / "bearing_data" / "2nd_test",
        "bearings": {
            "bearing1": ["bearing1_ch1"],
            "bearing2": ["bearing2_ch1"],
            "bearing3": ["bearing3_ch1"],
            "bearing4": ["bearing4_ch1"],
        },
        "known_failures": {
            "bearing1": "Outer Race Failure",
        }
    },
    "lathe_3": {
        "data_dir": BASE_DIR / "bearing_data" / "3rd_test" / "txt",
        "bearings": {
            "bearing1": ["bearing1_ch1"],
            "bearing2": ["bearing2_ch1"],
            "bearing3": ["bearing3_ch1"],
            "bearing4": ["bearing4_ch1"],
        },
        "known_failures": {
            "bearing3": "Outer Race Failure",
        }
    },
}

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

TRAIN_SPLIT = 0.2

# Alert threshold - consecutive anomalies before triggering CRITICAL alert.
# Raised to 50 to filter false positives caused by fault propagation from
# adjacent failing bearings increasing overall shaft vibration near end of test.
ANOMALY_STREAK_THRESHOLD = 50

# Delay between readings (0 = full speed, 0.1 = demo mode)
DELAY = 0.0


# ============ FEATURE EXTRACTION ============

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
    averaged = {}
    for key in all_channel_features[0].keys():
        averaged[key] = np.mean([f[key] for f in all_channel_features])
    return averaged


# ============ MODEL LOADING ============

def load_bearing_models(lathe_name, bearings):
    models = {}
    for bearing_name in bearings.keys():
        model_path = RESULTS_DIR / f"{lathe_name}_{bearing_name}_model.pkl"
        if not model_path.exists():
            print(f"ERROR: Model not found at {model_path}")
            print("Run train_isolation_forest.py first.")
            sys.exit(1)
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        models[bearing_name] = bundle
        print(f"  Loaded model: {model_path.name}")
    return models


# ============ HEALTH STATUS ============

def get_health_status(score, streak, latched):
    # If latched CRITICAL, stay CRITICAL regardless of current reading
    if latched:
        return "CRITICAL"
    if streak >= ANOMALY_STREAK_THRESHOLD:
        return "CRITICAL"
    elif score < 0:
        return "WARNING"
    else:
        return "HEALTHY"


def get_machine_status(bearing_statuses):
    if "CRITICAL" in bearing_statuses.values():
        return "CRITICAL"
    elif "WARNING" in bearing_statuses.values():
        return "WARNING"
    return "HEALTHY"


STATUS_ICONS = {
    "HEALTHY":  "✅",
    "WARNING":  "⚠️ ",
    "CRITICAL": "🔴",
}


# ============ RABBITMQ PUBLISH (placeholder) ============

def publish_to_rabbitmq(lathe_name, reading, bearing_data, machine_status):
    # TODO: Wire up RabbitMQ when integrating into 02_system
    # import pika
    # connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    # channel = connection.channel()
    # channel.queue_declare(queue=lathe_name)
    # message = json.dumps({...})
    # channel.basic_publish(exchange='', routing_key=lathe_name, body=message)
    pass


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='Per-bearing bearing monitor simulation')
    parser.add_argument('--lathe', type=str, default='lathe_1',
                        choices=['lathe_1', 'lathe_2', 'lathe_3'],
                        help='Which lathe to simulate')
    args = parser.parse_args()

    lathe_name = args.lathe
    config = LATHE_CONFIG[lathe_name]
    data_dir = config["data_dir"]
    bearings = config["bearings"]

    print(f"\n{'='*60}")
    print(f"BEARING ANOMALY DETECTION - PER-BEARING STREAM")
    print(f"{'='*60}")
    print(f"Lathe:   {lathe_name.upper()}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bearings monitored: {len(bearings)}")
    print("Press Ctrl+C to stop\n")

    # Load all bearing models
    print("Loading models...")
    models = load_bearing_models(lathe_name, bearings)
    print()

    # Get test files - skip training portion
    all_files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    split_idx = int(len(all_files) * TRAIN_SPLIT)
    test_files = all_files[split_idx:]

    print(f"Skipping first {split_idx} files (training data)")
    print(f"Streaming {len(test_files)} unseen files\n")

    # Per-bearing tracking
    bearing_streaks = {b: 0 for b in bearings}
    bearing_alerts = {b: False for b in bearings}
    bearing_latched = {b: False for b in bearings}  # latching - once CRITICAL stays CRITICAL
    bearing_alert_readings = {b: None for b in bearings}
    first_anomaly_readings = {b: None for b in bearings}

    total = len(test_files)

    try:
        for i, filepath in enumerate(test_files):
            reading_num = i + 1
            bearing_results = {}
            bearing_statuses = {}

            # Run per-bearing inference
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

                # Update streak (only if not already latched)
                if not bearing_latched[bearing_name]:
                    if is_anomaly:
                        bearing_streaks[bearing_name] += 1
                        if first_anomaly_readings[bearing_name] is None:
                            first_anomaly_readings[bearing_name] = reading_num
                    else:
                        bearing_streaks[bearing_name] = 0

                streak = bearing_streaks[bearing_name]

                # Trigger alert and latch if streak threshold hit
                if streak >= ANOMALY_STREAK_THRESHOLD and not bearing_alerts[bearing_name]:
                    bearing_alerts[bearing_name] = True
                    bearing_latched[bearing_name] = True
                    bearing_alert_readings[bearing_name] = reading_num
                    print(f"\n{'!'*60}")
                    print(f"ALERT: {bearing_name.upper()} on {lathe_name.upper()}")
                    print(f"  {ANOMALY_STREAK_THRESHOLD} consecutive anomalies detected!")
                    print(f"  Status LATCHED at CRITICAL - manual reset required")
                    print(f"  Schedule maintenance - bearing degradation likely")
                    print(f"{'!'*60}\n")

                status = get_health_status(score, streak, bearing_latched[bearing_name])
                bearing_results[bearing_name] = {
                    'score': score,
                    'is_anomaly': is_anomaly,
                    'kurtosis': features.get('kurtosis', 0),
                    'rms': features.get('rms', 0),
                    'streak': streak,
                    'status': status
                }
                bearing_statuses[bearing_name] = status

            # Overall machine status
            machine_status = get_machine_status(bearing_statuses)

            # Print reading summary
            print(f"[{reading_num}/{total}] {filepath.name} | Machine: {STATUS_ICONS[machine_status]} {machine_status}")
            for bearing_name, result in bearing_results.items():
                icon = STATUS_ICONS[result['status']]
                print(f"  {bearing_name}: {icon} {result['status']:8s} | "
                      f"score: {result['score']:+.3f} | "
                      f"kurt: {result['kurtosis']:6.2f} | "
                      f"rms: {result['rms']:.3f} | "
                      f"streak: {result['streak']}")

            # Publish to RabbitMQ
            publish_to_rabbitmq(lathe_name, reading_num, bearing_results, machine_status)

            if DELAY > 0:
                time.sleep(DELAY)

    except KeyboardInterrupt:
        print("\n\nStopped early by user")

    # ============ SUMMARY ============

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

    for bearing_name in bearings:
        print(f"\n{bearing_name.upper()}:")
        if first_anomaly_readings[bearing_name]:
            print(f"  First anomaly:   Reading {first_anomaly_readings[bearing_name]}")
        else:
            print(f"  First anomaly:   None detected")

        if bearing_alerts[bearing_name]:
            alert_r = bearing_alert_readings[bearing_name]
            readings_remaining = total - alert_r
            hours_warning = (readings_remaining * 10) / 60
            days_warning = hours_warning / 24
            print(f"  Alert triggered: Reading {alert_r} (LATCHED at CRITICAL)")
            print(f"  Warning time:    ~{hours_warning:.1f} hours (~{days_warning:.1f} days)")
        else:
            print(f"  Alert triggered: No (threshold: {ANOMALY_STREAK_THRESHOLD} consecutive)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
