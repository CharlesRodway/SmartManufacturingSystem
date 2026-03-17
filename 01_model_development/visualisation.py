# Visualisation Script for Training Results
# Digital Systems Project - Charles Rodway
#
# Reads results CSVs produced by train_isolation_forest.py and generates
# a separate 4-panel analysis plot for each test case (lathe_1, lathe_2, lathe_3).
#
# Channel mapping per test:
#
#   1st_test (lathe_1): 8 channels - 2 accelerometers per bearing
#     Failing: Bearing 3 (inner race) - bearing3_ch1, bearing3_ch2
#              Bearing 4 (rolling element) - bearing4_ch1, bearing4_ch2
#     Healthy: Bearing 1 - bearing1_ch1
#
#   2nd_test (lathe_2): 4 channels - 1 accelerometer per bearing
#     Columns map as: bearing1_ch1, bearing2_ch1, bearing3_ch1, bearing4_ch1
#     Failing: Bearing 1 (outer race) - bearing1_ch1
#     Healthy: Bearing 2 - bearing2_ch1
#
#   3rd_test (lathe_3): 4 channels - 1 accelerometer per bearing
#     Columns map as: bearing1_ch1, bearing2_ch1, bearing3_ch1, bearing4_ch1
#     Failing: Bearing 3 (outer race) - bearing3_ch1
#     Healthy: Bearing 1 - bearing1_ch1
#
# Output:
#   results/lathe_1_analysis.png
#   results/lathe_2_analysis.png
#   results/lathe_3_analysis.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============ SETTINGS ============

RESULTS_DIR = Path(__file__).resolve().parent / "results"

TRAIN_SPLIT = 0.2

TEST_CONFIG = {
    "lathe_1": {
        "title": "1st Test — Bearing 3: Inner Race Defect, Bearing 4: Rolling Element Defect",
        "kurtosis_bearings": [
            ("bearing3_ch1_kurtosis", "Bearing 3 ch1 (inner race)"),
            ("bearing3_ch2_kurtosis", "Bearing 3 ch2 (inner race)"),
            ("bearing4_ch1_kurtosis", "Bearing 4 ch1 (rolling element)"),
            ("bearing4_ch2_kurtosis", "Bearing 4 ch2 (rolling element)"),
        ],
        "rms_bearings": [
            ("bearing3_ch1_rms", "Bearing 3 (failing)", "solid"),
            ("bearing4_ch1_rms", "Bearing 4 (failing)", "solid"),
            ("bearing1_ch1_rms", "Bearing 1 (healthy)", "dashed"),
        ],
        "summary_kurtosis_col": "bearing3_ch1_kurtosis",
    },
    "lathe_2": {
        "title": "2nd Test — Bearing 1: Outer Race Failure",
        "kurtosis_bearings": [
            ("bearing1_ch1_kurtosis", "Bearing 1 (outer race failure)"),
            ("bearing2_ch1_kurtosis", "Bearing 2 (healthy)"),
        ],
        "rms_bearings": [
            ("bearing1_ch1_rms", "Bearing 1 (failing)", "solid"),
            ("bearing2_ch1_rms", "Bearing 2 (healthy)", "dashed"),
        ],
        "summary_kurtosis_col": "bearing1_ch1_kurtosis",
    },
    "lathe_3": {
        "title": "3rd Test — Bearing 3: Outer Race Failure",
        "kurtosis_bearings": [
            ("bearing3_ch1_kurtosis", "Bearing 3 (outer race failure)"),
            ("bearing1_ch1_kurtosis", "Bearing 1 (healthy)"),
        ],
        "rms_bearings": [
            ("bearing3_ch1_rms", "Bearing 3 (failing)", "solid"),
            ("bearing1_ch1_rms", "Bearing 1 (healthy)", "dashed"),
        ],
        "summary_kurtosis_col": "bearing3_ch1_kurtosis",
    },
}


# ============ PLOT ONE TEST CASE ============

def plot_results(name):
    csv_path = RESULTS_DIR / f"{name}_results.csv"

    if not csv_path.exists():
        print(f"WARNING: No results file found for {name} at {csv_path}")
        print(f"Run train_isolation_forest.py first.")
        return

    print(f"\nGenerating plot for {name.upper()}...")
    df = pd.read_csv(csv_path)
    config = TEST_CONFIG[name]

    df['progress'] = (df.index / len(df)) * 100
    cutoff_pct = TRAIN_SPLIT * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Bearing Anomaly Detection Results - NASA IMS Dataset ({name.upper()})\n{config["title"]}',
        fontsize=12, fontweight='bold'
    )

    # ---- Plot 1: Anomaly rate over time ----
    ax1 = axes[0, 0]
    df['rolling_anomaly'] = df['is_anomaly'].rolling(window=50, center=True).mean() * 100
    ax1.fill_between(df['progress'], df['rolling_anomaly'], alpha=0.3, color='red')
    ax1.plot(df['progress'], df['rolling_anomaly'], color='red', linewidth=1)
    anomalies = df[df['is_anomaly'] == True]
    ax1.scatter(anomalies['progress'], [2]*len(anomalies), marker='|',
                color='red', s=30, alpha=0.5)
    ax1.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax1.annotate('Training\ncutoff', xy=(cutoff_pct, 50),
                 xytext=(cutoff_pct + 2, 55), fontsize=8, color='blue')
    ax1.set_xlabel('Test Progress (%)')
    ax1.set_ylabel('Anomaly Rate (%)')
    ax1.set_title('Anomaly Detection Over Time')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # ---- Plot 2: Kurtosis - correct bearings per test ----
    ax2 = axes[0, 1]
    plotted_any = False
    for col, label in config["kurtosis_bearings"]:
        if col in df.columns:
            ax2.plot(df['progress'], df[col], label=label, alpha=0.7, linewidth=0.8)
            plotted_any = True
    if not plotted_any:
        ax2.text(0.5, 0.5, 'No kurtosis data available\nfor this test case',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=10)
    ax2.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Normal kurtosis ~0 (Fisher)')
    ax2.set_xlabel('Test Progress (%)')
    ax2.set_ylabel('Kurtosis')
    ax2.set_title('Kurtosis Over Time (Failing Bearings)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: Anomaly scores ----
    ax3 = axes[1, 0]
    normal = df[df['is_anomaly'] == False]
    anomalous = df[df['is_anomaly'] == True]
    ax3.scatter(normal['progress'], normal['anomaly_score'],
                c='blue', alpha=0.3, s=5, label='Normal')
    ax3.scatter(anomalous['progress'], anomalous['anomaly_score'],
                c='red', alpha=0.5, s=5, label='Anomaly')
    ax3.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax3.set_xlabel('Test Progress (%)')
    ax3.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax3.set_title('Anomaly Scores Over Time')
    ax3.legend(loc='lower left')
    ax3.set_xlim(0, 100)
    ax3.grid(True, alpha=0.3)

    # ---- Plot 4: RMS energy - correct bearings per test ----
    ax4 = axes[1, 1]
    for col, label, ls in config["rms_bearings"]:
        if col in df.columns:
            ax4.plot(df['progress'], df[col], label=label,
                     alpha=0.7, linewidth=0.8, linestyle=ls)
    ax4.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax4.set_xlabel('Test Progress (%)')
    ax4.set_ylabel('RMS (vibration energy)')
    ax4.set_title('RMS Energy Levels')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_xlim(0, 100)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / f"{name}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")

    # ---- Print summary stats ----
    total = len(df)
    num_anomalies = df['is_anomaly'].sum()
    summary_col = config["summary_kurtosis_col"]
    print(f"\nSummary for {name.upper()}:")
    print(f"  Total samples:  {total}")
    print(f"  Anomalies:      {num_anomalies} ({100*num_anomalies/total:.1f}%)")
    if summary_col in df.columns:
        print(f"  Max kurtosis ({summary_col}): {df[summary_col].max():.2f}")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("Visualisation - All Test Cases")
    print("=" * 50)

    for name in TEST_CONFIG.keys():
        plot_results(name)

    print(f"\n{'='*50}")
    print("All plots generated.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
