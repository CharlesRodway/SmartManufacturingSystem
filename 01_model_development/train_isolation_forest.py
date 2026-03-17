# Train Isolation Forest Model for Bearing Anomaly Detection
# Digital Systems Project - Charles Rodway
#
# Trains a separate Isolation Forest model for each NASA IMS test case.
# Each model is trained on the first 20% of its dataset (healthy period)
# and saved as lathe_1, lathe_2, lathe_3 respectively.
# Results for each test are saved separately for visualisation.
#
# Dataset: NASA IMS Bearing Dataset (1st_test, 2nd_test, 3rd_test)
#
# Channel mapping:
#   1st_test: 8 channels (2 accelerometers per bearing)
#     bearing1_ch1, bearing1_ch2, bearing2_ch1, bearing2_ch2,
#     bearing3_ch1, bearing3_ch2, bearing4_ch1, bearing4_ch2
#
#   2nd_test / 3rd_test: 4 channels (1 accelerometer per bearing)
#     bearing1_ch1, bearing2_ch1, bearing3_ch1, bearing4_ch1

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
    {"name": "lathe_1", "data_dir": BASE_DIR / "bearing_data" / "1st_test"},
    {"name": "lathe_2", "data_dir": BASE_DIR / "bearing_data" / "2nd_test"},
    {"name": "lathe_3", "data_dir": BASE_DIR / "bearing_data" / "3rd_test" / "txt"},
]

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

TRAIN_SPLIT = 0.2

# Column names for each channel count
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
    # Kurtosis is the main indicator for bearing faults -
    # healthy bearings sit around 3, faulty ones spike way higher.
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
    # Detect number of columns and assign correct names
    df = pd.read_csv(filepath, sep='\t', header=None)
    n_cols = df.shape[1]

    if n_cols == 8:
        df.columns = COLS_8
    elif n_cols == 4:
        df.columns = COLS_4
    else:
        # Fallback - just number them
        df.columns = [f'ch{i}' for i in range(n_cols)]

    return df


def extract_features_from_file(filepath):
    df = load_bearing_file(filepath)
    all_features = {}
    for column in df.columns:
        features = calculate_features(df[column].values)
        for name, val in features.items():
            all_features[f'{column}_{name}'] = val
    return all_features


# ============ TRAIN ONE MODEL ============

def train_model(test_case):
    name = test_case["name"]
    data_dir = test_case["data_dir"]

    print(f"\n{'='*50}")
    print(f"Training model for {name.upper()} ({data_dir.name})")
    print(f"{'='*50}")

    if not data_dir.exists():
        print(f"WARNING: Data directory not found: {data_dir}")
        print(f"Skipping {name}.")
        return

    # Get all data files sorted by timestamp
    all_files = sorted([f for f in data_dir.iterdir() if f.is_file()])
    total_files = len(all_files)

    if total_files == 0:
        print(f"WARNING: No files found in {data_dir}. Skipping {name}.")
        return

    split_idx = int(total_files * TRAIN_SPLIT)
    train_files = all_files[:split_idx]

    # Detect channel count from first file
    sample_df = pd.read_csv(all_files[0], sep='\t', header=None)
    n_channels = sample_df.shape[1]
    col_names = COLS_8 if n_channels == 8 else COLS_4

    print(f"Total files:       {total_files}")
    print(f"Channels per file: {n_channels} ({len(col_names)} bearings/channels)")
    print(f"Training on:       {len(train_files)} files (first {TRAIN_SPLIT*100:.0f}%)")
    print(f"Reserved for test: {total_files - len(train_files)} files")

    # Extract features from training files
    print("\nExtracting features...")
    train_data = []
    for i, filepath in enumerate(train_files):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(train_files)}...")
        train_data.append(extract_features_from_file(filepath))

    df_train = pd.DataFrame(train_data)
    feature_names = list(df_train.columns)
    X = df_train.values

    print(f"Training data shape: {X.shape}")

    # Train model
    print("\nTraining Isolation Forest...")
    model = IsolationForest(
        contamination=0.01,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / f"{name}_model.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'train_split': TRAIN_SPLIT,
            'n_train_samples': len(train_files),
            'n_channels': n_channels
        }, f)

    print(f"Model saved to {model_path}")

    # Run inference on ALL files and save results for visualisation
    print("\nRunning inference on full dataset for visualisation...")
    all_results = []

    for i, filepath in enumerate(all_files):
        features = extract_features_from_file(filepath)
        X_pred = np.array([[features[n] for n in feature_names]])
        pred = model.predict(X_pred)[0]
        score = model.decision_function(X_pred)[0]
        is_anomaly = (pred == -1)

        row = {'reading': i + 1, 'filename': filepath.name,
               'is_anomaly': is_anomaly, 'anomaly_score': score}
        for feat_name, val in features.items():
            row[feat_name] = val

        all_results.append(row)

    results_df = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / f"{name}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    print(f"Done - {name.upper()}")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("Isolation Forest Training - All Test Cases")
    print("=" * 50)

    for test_case in TEST_CASES:
        train_model(test_case)

    print(f"\n{'='*50}")
    print("All models trained successfully.")
    print(f"Models and results saved to: {OUTPUT_DIR}")
    print("Run visualisation.py to generate plots.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
