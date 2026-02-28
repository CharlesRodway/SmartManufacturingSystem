# Train Isolation Forest Models for Bearing Anomaly Detection
# Digital Systems Project - Charles Rodway
#
# Trains one Isolation Forest model per bearing per test case.
# Each model is trained only on its own bearing's features (11 features)
# so it learns what *that specific bearing* looks like when healthy.
#
# Models saved as: results/lathe_X_bearingY_model.pkl
# Results saved as: results/lathe_X_bearingY_results.csv
# Overall results: results/lathe_X_overall_results.csv
#
# Channel mapping:
#   1st_test: 8 channels (2 accelerometers per bearing)
#     Bearings 1-4, each with ch1 and ch2
#     Features extracted per channel, then averaged per bearing
#
#   2nd_test / 3rd_test: 4 channels (1 accelerometer per bearing)
#     Bearings 1-4, each with ch1 only

import os
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from pathlib import Path
from sklearn.ensemble import IsolationForest


# ============ SETTINGS ============

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CASES = [
    {
        "name": "lathe_1",
        "data_dir": BASE_DIR / "bearing_data" / "1st_test",
        "bearings": {
            "bearing1": ["bearing1_ch1", "bearing1_ch2"],
            "bearing2": ["bearing2_ch1", "bearing2_ch2"],
            "bearing3": ["bearing3_ch1", "bearing3_ch2"],
            "bearing4": ["bearing4_ch1", "bearing4_ch2"],
        },
        "known_failures": {
            "bearing3": "Inner race defect",
            "bearing4": "Rolling element defect",
        }
    },
    {
        "name": "lathe_2",
        "data_dir": BASE_DIR / "bearing_data" / "2nd_test",
        "bearings": {
            "bearing1": ["bearing1_ch1"],
            "bearing2": ["bearing2_ch1"],
            "bearing3": ["bearing3_ch1"],
            "bearing4": ["bearing4_ch1"],
        },
        "known_failures": {
            "bearing1": "Outer race failure",
        }
    },
    {
        "name": "lathe_3",
        "data_dir": BASE_DIR / "bearing_data" / "3rd_test" / "txt",
        "bearings": {
            "bearing1": ["bearing1_ch1"],
            "bearing2": ["bearing2_ch1"],
            "bearing3": ["bearing3_ch1"],
            "bearing4": ["bearing4_ch1"],
        },
        "known_failures": {
            "bearing3": "Outer race failure",
        }
    },
]

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

TRAIN_SPLIT = 0.2

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


def load_bearing_file(filepath):
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
    # Extract features for a specific bearing's channels
    # If bearing has 2 channels, average the features across both
    df = load_bearing_file(filepath)
    all_channel_features = []

    for ch in channels:
        if ch in df.columns:
            feats = calculate_features(df[ch].values)
            all_channel_features.append(feats)

    if not all_channel_features:
        return None

    # Average across channels if multiple
    averaged = {}
    for key in all_channel_features[0].keys():
        averaged[key] = np.mean([f[key] for f in all_channel_features])

    return averaged


# ============ TRAIN ONE BEARING MODEL ============

def train_bearing_model(lathe_name, bearing_name, channels, all_files, train_files):
    print(f"  Training {bearing_name}... ({len(channels)} channel(s))")

    # Extract features for training files
    train_data = []
    for filepath in train_files:
        feats = extract_bearing_features(filepath, channels)
        if feats:
            train_data.append(feats)

    if not train_data:
        print(f"  WARNING: No data extracted for {bearing_name}, skipping.")
        return None, None

    df_train = pd.DataFrame(train_data)
    feature_names = list(df_train.columns)
    X = df_train.values

    # Train model
    model = IsolationForest(
        contamination=0.01,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)

    # Save model
    model_path = OUTPUT_DIR / f"{lathe_name}_{bearing_name}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'train_split': TRAIN_SPLIT,
            'n_train_samples': len(train_files),
            'channels': channels,
            'bearing': bearing_name,
            'lathe': lathe_name
        }, f)

    print(f"    Model saved: {model_path.name}")

    # Run inference on ALL files
    results = []
    for i, filepath in enumerate(all_files):
        feats = extract_bearing_features(filepath, channels)
        if feats:
            X_pred = np.array([[feats[n] for n in feature_names]])
            pred = model.predict(X_pred)[0]
            score = model.decision_function(X_pred)[0]
            is_anomaly = (pred == -1)

            row = {
                'reading': i + 1,
                'filename': filepath.name,
                'is_anomaly': is_anomaly,
                'anomaly_score': score,
            }
            for feat_name, val in feats.items():
                row[feat_name] = val

            results.append(row)

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / f"{lathe_name}_{bearing_name}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"    Results saved: {results_path.name}")

    return model, results_df


# ============ TRAIN ONE LATHE (ALL BEARINGS) ============

def train_lathe(test_case):
    name = test_case["name"]
    data_dir = test_case["data_dir"]
    bearings = test_case["bearings"]

    print(f"\n{'='*50}")
    print(f"Training models for {name.upper()} ({data_dir.name})")
    print(f"{'='*50}")

    if not data_dir.exists():
        print(f"WARNING: Data directory not found: {data_dir}. Skipping.")
        return

    all_files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    total_files = len(all_files)

    if total_files == 0:
        print(f"WARNING: No files found in {data_dir}. Skipping.")
        return

    split_idx = int(total_files * TRAIN_SPLIT)
    train_files = all_files[:split_idx]

    sample_df = pd.read_csv(all_files[0], sep='\t', header=None)
    n_channels = sample_df.shape[1]

    print(f"Total files:       {total_files}")
    print(f"Channels per file: {n_channels}")
    print(f"Training on:       {len(train_files)} files (first {TRAIN_SPLIT*100:.0f}%)")
    print(f"Reserved for test: {total_files - len(train_files)} files")
    print(f"Training {len(bearings)} individual bearing models...\n")

    # Train a model for each bearing
    all_bearing_results = {}
    for bearing_name, channels in bearings.items():
        model, results_df = train_bearing_model(
            name, bearing_name, channels, all_files, train_files
        )
        if results_df is not None:
            all_bearing_results[bearing_name] = results_df

    # Build overall results combining all bearings
    if all_bearing_results:
        print(f"\n  Building overall machine health results...")
        n_readings = len(all_files)
        overall_rows = []

        for i in range(n_readings):
            row = {'reading': i + 1}
            bearing_scores = []
            bearing_anomalies = []

            for bearing_name, results_df in all_bearing_results.items():
                if i < len(results_df):
                    score = results_df.iloc[i]['anomaly_score']
                    is_anomaly = results_df.iloc[i]['is_anomaly']
                    row[f'{bearing_name}_score'] = score
                    row[f'{bearing_name}_anomaly'] = is_anomaly
                    row[f'{bearing_name}_kurtosis'] = results_df.iloc[i].get('kurtosis', 0)
                    row[f'{bearing_name}_rms'] = results_df.iloc[i].get('rms', 0)
                    bearing_scores.append(score)
                    bearing_anomalies.append(is_anomaly)

            # Overall machine health = worst bearing score (lowest = most anomalous)
            row['machine_score'] = min(bearing_scores) if bearing_scores else 0
            row['machine_anomaly'] = any(bearing_anomalies)
            row['anomalous_bearing_count'] = sum(bearing_anomalies)
            overall_rows.append(row)

        overall_df = pd.DataFrame(overall_rows)
        overall_path = OUTPUT_DIR / f"{name}_overall_results.csv"
        overall_df.to_csv(overall_path, index=False)
        print(f"  Overall results saved: {overall_path.name}")

    print(f"\nDone - {name.upper()}")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("Per-Bearing Isolation Forest Training - All Lathes")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for test_case in TEST_CASES:
        train_lathe(test_case)

    print(f"\n{'='*50}")
    print("All models trained successfully.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Run visualisation.py to generate plots.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
