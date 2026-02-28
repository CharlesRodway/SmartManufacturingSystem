# Trainer Service
# Digital Systems Project - Charles Rodway
#
# Watches lathes.json for lathes in "collecting" status.
# When enough data has been collected AND engineer has confirmed
# the machine is healthy, auto-trains per-bearing Isolation Forest models
# and switches the lathe to "monitoring" status.
#
# Runs as a background service, checking every 60 seconds.

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest


# ============ SETTINGS ============

CONFIG_PATH = Path("/app/config/lathes.json")
MODELS_DIR = Path("/app/models")
DATA_DIR = Path("/app/data")

TRAIN_SPLIT = 0.2
CHECK_INTERVAL = 60  # seconds between checks

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


# ============ FEATURE EXTRACTION ============

def calculate_features(signal):
    signal = np.array(signal, dtype=np.float64)
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'rms': np.sqrt(np.mean(signal ** 2)),
        'peak_to_peak': np.max(signal) - np.min(signal),
        'crest_factor': np.max(signal) / np.sqrt(np.mean(signal ** 2)) if np.sqrt(np.mean(signal ** 2)) != 0 else 0,
        'kurtosis': stats.kurtosis(signal, fisher=True),
        'skewness': stats.skew(signal),
        'shape_factor': np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0,
        'impulse_factor': np.max(signal) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
    }


def load_file(filepath, n_channels):
    df = pd.read_csv(filepath, sep='\t', header=None)
    if df.shape[1] == 8:
        df.columns = COLS_8
    elif df.shape[1] == 4:
        df.columns = COLS_4
    else:
        df.columns = [f'ch{i}' for i in range(df.shape[1])]
    return df


def extract_bearing_features(filepath, channels, n_channels):
    df = load_file(filepath, n_channels)
    all_channel_features = []
    for ch in channels:
        if ch in df.columns:
            all_channel_features.append(calculate_features(df[ch].values))
    if not all_channel_features:
        return None
    averaged = {}
    for key in all_channel_features[0].keys():
        averaged[key] = np.mean([f[key] for f in all_channel_features])
    return averaged


# ============ TRAINING ============

def train_lathe_models(lathe_name, config):
    print(f"\nTraining models for {lathe_name.upper()}...")
    bearings = config["bearings"]
    sensors_per_bearing = config["sensors_per_bearing"]
    data_dir = DATA_DIR / lathe_name

    all_files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    if not all_files:
        print(f"  ERROR: No data files found in {data_dir}")
        return False

    split_idx = int(len(all_files) * TRAIN_SPLIT)
    train_files = all_files[:split_idx]

    if not train_files:
        print(f"  ERROR: Not enough files for training")
        return False

    print(f"  Total files: {len(all_files)}")
    print(f"  Training on: {len(train_files)} files (first {TRAIN_SPLIT*100:.0f}%)")

    # Create model output directory
    model_dir = MODELS_DIR / lathe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for bearing_name, channels in bearings.items():
        print(f"  Training {bearing_name}...")

        train_data = []
        for filepath in train_files:
            feats = extract_bearing_features(filepath, channels, sensors_per_bearing)
            if feats:
                train_data.append(feats)

        if not train_data:
            print(f"    WARNING: No data extracted for {bearing_name}")
            continue

        df_train = pd.DataFrame(train_data)
        feature_names = list(df_train.columns)
        X = df_train.values

        model = IsolationForest(
            contamination=0.01,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)

        model_path = model_dir / f"{bearing_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': feature_names,
                'train_split': TRAIN_SPLIT,
                'channels': channels,
                'bearing': bearing_name,
                'lathe': lathe_name,
                'trained_at': datetime.now().isoformat()
            }, f)

        print(f"    Saved: {model_path.name}")

    print(f"Training complete for {lathe_name.upper()}")
    return True


# ============ CONFIG HELPERS ============

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def update_lathe_status(lathe_name, status):
    config = load_config()
    config[lathe_name]["status"] = status
    save_config(config)
    print(f"  {lathe_name} status updated to: {status}")


# ============ MAIN LOOP ============

def main():
    print(f"\n{'='*60}")
    print("CNC PREDICTIVE MAINTENANCE - TRAINER SERVICE")
    print(f"{'='*60}")
    print(f"Watching for new lathes every {CHECK_INTERVAL}s...")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Data dir:   {DATA_DIR}\n")

    while True:
        try:
            config = load_config()

            for lathe_name, lathe_config in config.items():
                status = lathe_config.get("status")

                # Only process lathes that are ready to train
                if status != "collecting":
                    continue

                hours_required = lathe_config.get("collection_hours_required", 336)
                hours_completed = lathe_config.get("collection_hours_completed", 0)
                healthy_confirmed = lathe_config.get("healthy_confirmed", False)

                print(f"[{lathe_name}] Collecting: {hours_completed}/{hours_required}h | "
                      f"Healthy confirmed: {healthy_confirmed}")

                # Check if ready to train
                if hours_completed >= hours_required and healthy_confirmed:
                    print(f"\n[{lathe_name}] Ready to train!")
                    update_lathe_status(lathe_name, "training")

                    success = train_lathe_models(lathe_name, lathe_config)

                    if success:
                        update_lathe_status(lathe_name, "monitoring")
                        print(f"[{lathe_name}] Now monitoring!")
                    else:
                        update_lathe_status(lathe_name, "training_failed")
                        print(f"[{lathe_name}] Training failed - check data")

        except Exception as e:
            print(f"Trainer error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
