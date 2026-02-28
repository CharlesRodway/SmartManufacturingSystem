# Visualisation Script for Per-Bearing Training Results
# Digital Systems Project - Charles Rodway
#
# Generates two types of plots for each lathe:
#
# 1. Per-bearing plots (4 panels each):
#    - Anomaly rate over time
#    - Kurtosis over time
#    - Anomaly scores over time
#    - RMS energy over time
#
# 2. Overall machine health summary plot:
#    - All bearing anomaly scores overlaid
#    - Overall machine health score
#    - Anomalous bearing count over time
#    - Per-bearing kurtosis comparison
#
# Output per lathe (e.g. lathe_1):
#   results/lathe_1_bearing1_analysis.png
#   results/lathe_1_bearing2_analysis.png
#   results/lathe_1_bearing3_analysis.png
#   results/lathe_1_bearing4_analysis.png
#   results/lathe_1_overall_analysis.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============ SETTINGS ============

RESULTS_DIR = Path(__file__).resolve().parent / "results"

TRAIN_SPLIT = 0.2

TEST_CONFIG = {
    "lathe_1": {
        "title": "1st Test",
        "bearings": ["bearing1", "bearing2", "bearing3", "bearing4"],
        "known_failures": {
            "bearing3": "Inner Race Defect",
            "bearing4": "Rolling Element Defect",
        },
    },
    "lathe_2": {
        "title": "2nd Test",
        "bearings": ["bearing1", "bearing2", "bearing3", "bearing4"],
        "known_failures": {
            "bearing1": "Outer Race Failure",
        },
    },
    "lathe_3": {
        "title": "3rd Test",
        "bearings": ["bearing1", "bearing2", "bearing3", "bearing4"],
        "known_failures": {
            "bearing3": "Outer Race Failure",
        },
    },
}

# Colours for each bearing - consistent across all plots
BEARING_COLOURS = {
    "bearing1": "#2196F3",  # blue
    "bearing2": "#4CAF50",  # green
    "bearing3": "#FF5722",  # red-orange
    "bearing4": "#9C27B0",  # purple
}


# ============ PER-BEARING PLOT ============

def plot_bearing(lathe_name, bearing_name, config):
    csv_path = RESULTS_DIR / f"{lathe_name}_{bearing_name}_results.csv"

    if not csv_path.exists():
        print(f"  WARNING: No results found for {bearing_name}, skipping.")
        return

    df = pd.read_csv(csv_path)
    df['progress'] = (df.index / len(df)) * 100
    cutoff_pct = TRAIN_SPLIT * 100
    colour = BEARING_COLOURS.get(bearing_name, "steelblue")

    known_failures = config.get("known_failures", {})
    failure_label = known_failures.get(bearing_name, "Healthy (no known failure)")
    is_failing = bearing_name in known_failures

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'{lathe_name.upper()} — {bearing_name.replace("b", "B").replace("earing", "earing ")} '
        f'({config["title"]})\n{failure_label}',
        fontsize=12, fontweight='bold'
    )

    # ---- Plot 1: Anomaly rate ----
    ax1 = axes[0, 0]
    df['rolling_anomaly'] = df['is_anomaly'].rolling(window=50, center=True).mean() * 100
    ax1.fill_between(df['progress'], df['rolling_anomaly'], alpha=0.3,
                     color='red' if is_failing else 'green')
    ax1.plot(df['progress'], df['rolling_anomaly'], linewidth=1,
             color='red' if is_failing else 'green')
    anomalies = df[df['is_anomaly'] == True]
    ax1.scatter(anomalies['progress'], [2]*len(anomalies), marker='|',
                color='red', s=30, alpha=0.5)
    ax1.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax1.annotate('Training cutoff', xy=(cutoff_pct, 50),
                 xytext=(cutoff_pct + 2, 55), fontsize=8, color='blue')
    ax1.set_xlabel('Test Progress (%)')
    ax1.set_ylabel('Anomaly Rate (%)')
    ax1.set_title('Anomaly Rate Over Time')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # ---- Plot 2: Kurtosis ----
    ax2 = axes[0, 1]
    if 'kurtosis' in df.columns:
        ax2.plot(df['progress'], df['kurtosis'], color=colour,
                 alpha=0.7, linewidth=0.8, label=bearing_name)
    ax2.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Normal kurtosis ~0 (Fisher)')
    ax2.set_xlabel('Test Progress (%)')
    ax2.set_ylabel('Kurtosis (Fisher)')
    ax2.set_title('Kurtosis Over Time')
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

    # ---- Plot 4: RMS ----
    ax4 = axes[1, 1]
    if 'rms' in df.columns:
        ax4.plot(df['progress'], df['rms'], color=colour,
                 alpha=0.7, linewidth=0.8, label=bearing_name)
    ax4.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax4.set_xlabel('Test Progress (%)')
    ax4.set_ylabel('RMS (vibration energy)')
    ax4.set_title('RMS Energy Over Time')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_xlim(0, 100)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / f"{lathe_name}_{bearing_name}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

    # Summary stats
    total = len(df)
    num_anomalies = df['is_anomaly'].sum()
    print(f"  {bearing_name}: {num_anomalies}/{total} anomalies ({100*num_anomalies/total:.1f}%)", end="")
    if 'kurtosis' in df.columns:
        print(f" | Max kurtosis: {df['kurtosis'].max():.2f}", end="")
    print()


# ============ OVERALL MACHINE HEALTH PLOT ============

def plot_overall(lathe_name, config):
    csv_path = RESULTS_DIR / f"{lathe_name}_overall_results.csv"

    if not csv_path.exists():
        print(f"  WARNING: No overall results found for {lathe_name}, skipping.")
        return

    df = pd.read_csv(csv_path)
    df['progress'] = (df.index / len(df)) * 100
    cutoff_pct = TRAIN_SPLIT * 100
    bearings = config["bearings"]
    known_failures = config.get("known_failures", {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'{lathe_name.upper()} — Overall Machine Health ({config["title"]})\n'
        f'Failed bearings: {", ".join([f"{b} ({v})" for b, v in known_failures.items()]) if known_failures else "None recorded"}',
        fontsize=12, fontweight='bold'
    )

    # ---- Plot 1: Overall machine health score ----
    ax1 = axes[0, 0]
    if 'machine_score' in df.columns:
        # Normalise score to 0-100% health (higher score = healthier)
        min_s = df['machine_score'].min()
        max_s = df['machine_score'].max()
        if max_s != min_s:
            health_pct = ((df['machine_score'] - min_s) / (max_s - min_s)) * 100
        else:
            health_pct = pd.Series([100] * len(df))
        ax1.fill_between(df['progress'], health_pct, alpha=0.2, color='blue')
        ax1.plot(df['progress'], health_pct, color='blue', linewidth=1, label='Machine health %')
    ax1.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax1.annotate('Training cutoff', xy=(cutoff_pct, 50),
                 xytext=(cutoff_pct + 2, 55), fontsize=8, color='blue')
    ax1.set_xlabel('Test Progress (%)')
    ax1.set_ylabel('Machine Health Score (%)')
    ax1.set_title('Overall Machine Health Over Time')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # ---- Plot 2: Per-bearing anomaly scores overlaid ----
    ax2 = axes[0, 1]
    for bearing in bearings:
        score_col = f'{bearing}_score'
        if score_col in df.columns:
            label = bearing.replace('bearing', 'Bearing ')
            if bearing in known_failures:
                label += f' ⚠ ({known_failures[bearing]})'
            ax2.plot(df['progress'], df[score_col],
                     color=BEARING_COLOURS.get(bearing, 'gray'),
                     alpha=0.7, linewidth=0.8, label=label)
    ax2.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax2.set_xlabel('Test Progress (%)')
    ax2.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax2.set_title('Per-Bearing Anomaly Scores')
    ax2.legend(loc='lower left', fontsize=7)
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: Number of anomalous bearings over time ----
    ax3 = axes[1, 0]
    if 'anomalous_bearing_count' in df.columns:
        rolling = df['anomalous_bearing_count'].rolling(window=50, center=True).mean()
        ax3.fill_between(df['progress'], rolling, alpha=0.3, color='orange')
        ax3.plot(df['progress'], rolling, color='orange', linewidth=1)
    ax3.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax3.set_xlabel('Test Progress (%)')
    ax3.set_ylabel('Number of Bearings Flagged')
    ax3.set_title('Anomalous Bearing Count Over Time')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, len(bearings))
    ax3.set_yticks(range(len(bearings) + 1))
    ax3.grid(True, alpha=0.3)

    # ---- Plot 4: Per-bearing kurtosis comparison ----
    ax4 = axes[1, 1]
    for bearing in bearings:
        kurt_col = f'{bearing}_kurtosis'
        if kurt_col in df.columns:
            label = bearing.replace('bearing', 'Bearing ')
            if bearing in known_failures:
                label += f' ⚠'
            ax4.plot(df['progress'], df[kurt_col],
                     color=BEARING_COLOURS.get(bearing, 'gray'),
                     alpha=0.7, linewidth=0.8, label=label)
    ax4.axvline(x=cutoff_pct, color='blue', linestyle='--', linewidth=1.2, alpha=0.8)
    ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Normal ~0')
    ax4.set_xlabel('Test Progress (%)')
    ax4.set_ylabel('Kurtosis (Fisher)')
    ax4.set_title('Per-Bearing Kurtosis Comparison')
    ax4.legend(loc='upper left', fontsize=7)
    ax4.set_xlim(0, 100)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / f"{lathe_name}_overall_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("Visualisation - Per-Bearing + Overall Machine Health")
    print("=" * 50)

    for lathe_name, config in TEST_CONFIG.items():
        print(f"\n--- {lathe_name.upper()} ---")

        # Per-bearing plots
        for bearing in config["bearings"]:
            plot_bearing(lathe_name, bearing, config)

        # Overall machine health plot
        plot_overall(lathe_name, config)

    print(f"\n{'='*50}")
    print("All plots generated.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
