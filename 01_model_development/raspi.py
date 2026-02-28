# Raspberry Pi Bearing Monitor - Live Inference Demo
# Digital Systems Project - Charles Rodway
#
# This streams the unseen portion of the dataset through the trained model
# Simulates what would happen on a real raspberry pi monitoring a CNC machine

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime


# ============ SETTINGS ============

MODEL_PATH = "models/isolation_forest_model.pkl"
DATA_DIR = "bearing_data/1st_test"

# this needs to match whatever i used in training
# trained on first 20% so we test on remaining 80%
TRAIN_SPLIT = 0.2

# delay between readings - set to 0 for full speed, 0.3 for demo
DELAY = 0.0

# for when i add rabbitmq later
MACHINE_ID = "CNC-LATHE-001"

# alert threshold - how many anomalies in a row before we trigger alert
ANOMALY_STREAK_THRESHOLD = 30


# ============ FEATURE EXTRACTION ============

def calc_features(signal):
    # same features as training - 11 statistical measures
    signal = np.array(signal, dtype=np.float64)
    
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    p2p = max_val - min_val
    crest = max_val / rms if rms != 0 else 0
    kurt = stats.kurtosis(signal, fisher=True)
    skew = stats.skew(signal)
    mean_abs = np.mean(np.abs(signal))
    shape = rms / mean_abs if mean_abs != 0 else 0
    impulse = max_val / mean_abs if mean_abs != 0 else 0
    
    return {
        'mean': mean_val, 'std': std_val, 'max': max_val, 'min': min_val,
        'rms': rms, 'peak_to_peak': p2p, 'crest_factor': crest,
        'kurtosis': kurt, 'skewness': skew,
        'shape_factor': shape, 'impulse_factor': impulse
    }


def load_file(filepath):
    # nasa ims format - 8 columns, tab separated
    cols = ['bearing1_ch1', 'bearing1_ch2', 'bearing2_ch1', 'bearing2_ch2',
            'bearing3_ch1', 'bearing3_ch2', 'bearing4_ch1', 'bearing4_ch2']
    return pd.read_csv(filepath, sep='\t', header=None, names=cols)


def extract_features(filepath):
    # get all 88 features from one file (11 features x 8 channels)
    df = load_file(filepath)
    features = {}
    
    for col in df.columns:
        col_feats = calc_features(df[col].values)
        for name, val in col_feats.items():
            features[f'{col}_{name}'] = val
    
    return features


# ============ MODEL STUFF ============

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['feature_names']


def predict(model, feature_names, features):
    X = np.array([[features[name] for name in feature_names]])
    pred = model.predict(X)[0]
    score = model.decision_function(X)[0]
    is_anomaly = (pred == -1)
    return is_anomaly, score


# ============ MAIN ============

def main():
    print(f"\n{'='*50}")
    print("BEARING ANOMALY DETECTION - STREAM DEMO")
    print(f"{'='*50}")
    print(f"Machine: {MACHINE_ID}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop\n")
    
    # check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: no model at {MODEL_PATH}")
        print("Run train_isolation_forest.py first")
        sys.exit(1)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: no data at {DATA_DIR}")
        sys.exit(1)
    
    # load model
    model, feature_names = load_model()
    print("Model loaded")
    
    # get files and skip training portion
    data_path = Path(DATA_DIR)
    all_files = sorted([f for f in data_path.iterdir() if f.is_file()])
    
    split_idx = int(len(all_files) * TRAIN_SPLIT)
    test_files = all_files[split_idx:]
    
    print(f"Skipping first {split_idx} files (used for training)")
    print(f"Streaming {len(test_files)} unseen files\n")
    
    # tracking variables
    total = len(test_files)
    anomaly_count = 0
    current_streak = 0
    max_streak = 0
    alert_triggered = False
    alert_reading = None
    alert_timestamp = None
    first_anomaly_reading = None
    first_anomaly_timestamp = None
    max_kurtosis = 0
    max_kurtosis_reading = None
    
    # for the summary at the end
    results = []
    
    try:
        for i, filepath in enumerate(test_files):
            # extract features and predict
            features = extract_features(filepath)
            is_anomaly, score = predict(model, feature_names, features)
            
            # track kurtosis (bearing 3 is the one that fails)
            kurt = features['bearing3_ch1_kurtosis']
            if kurt > max_kurtosis:
                max_kurtosis = kurt
                max_kurtosis_reading = i + 1
            
            if is_anomaly:
                anomaly_count += 1
                current_streak += 1
                status = "ANOMALY"
                
                # track first anomaly
                if first_anomaly_reading is None:
                    first_anomaly_reading = i + 1
                    first_anomaly_timestamp = filepath.name
                
                # track max streak
                if current_streak > max_streak:
                    max_streak = current_streak
                
                # check if we should trigger alert
                if current_streak >= ANOMALY_STREAK_THRESHOLD and not alert_triggered:
                    alert_triggered = True
                    alert_reading = i + 1
                    alert_timestamp = filepath.name
                    print(f"\n{'!'*50}")
                    print(f"ALERT: {ANOMALY_STREAK_THRESHOLD} consecutive anomalies detected!")
                    print(f"Bearing degradation likely - schedule maintenance")
                    print(f"{'!'*50}\n")
            else:
                current_streak = 0
                status = "OK"
            
            # store for summary
            results.append({
                'reading': i + 1,
                'timestamp': filepath.name,
                'is_anomaly': is_anomaly,
                'score': score,
                'kurtosis': kurt
            })
            
            # print status
            print(f"[{i+1}/{total}] {filepath.name} | {status} | score: {score:.3f} | kurtosis: {kurt:.2f}")
            
            if DELAY > 0:
                time.sleep(DELAY)
            
    except KeyboardInterrupt:
        print("\n\nStopped early by user")
    
    # ============ SUMMARY ============
    
    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*50}")
    
    readings_processed = len(results)
    anomaly_pct = (anomaly_count / readings_processed * 100) if readings_processed > 0 else 0
    
    print(f"\nTotal readings processed: {readings_processed}")
    print(f"Total anomalies:          {anomaly_count} ({anomaly_pct:.1f}%)")
    print(f"Max anomaly streak:       {max_streak}")
    print(f"Max kurtosis:             {max_kurtosis:.2f} (reading {max_kurtosis_reading})")
    
    print(f"\n--- Timeline ---")
    if first_anomaly_reading:
        print(f"First anomaly:      Reading {first_anomaly_reading} ({first_anomaly_timestamp})")
    
    if alert_triggered:
        print(f"Alert triggered:    Reading {alert_reading} ({alert_timestamp})")
        
        # calculate early warning time
        # each reading is roughly 10 mins apart in the nasa dataset
        readings_before_end = readings_processed - alert_reading
        hours_warning = (readings_before_end * 10) / 60
        days_warning = hours_warning / 24
        
        print(f"\n--- Early Warning Analysis ---")
        print(f"Readings after alert:     {readings_before_end}")
        print(f"Estimated warning time:   {hours_warning:.1f} hours (~{days_warning:.1f} days)")
    else:
        print(f"Alert triggered:    No (threshold: {ANOMALY_STREAK_THRESHOLD} consecutive)")
    
    # final recommendation
    print(f"\n--- Recommendation ---")
    if alert_triggered:
        print("STATUS: MAINTENANCE REQUIRED")
        print("Consistent anomaly pattern detected indicating bearing degradation.")
        print(f"System provided ~{days_warning:.1f} days advance warning before end of test.")
    elif anomaly_pct > 20:
        print("STATUS: MONITOR CLOSELY")
        print("Elevated anomaly rate detected. Recommend increased monitoring frequency.")
    else:
        print("STATUS: HEALTHY")
        print("No significant anomaly patterns detected.")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()