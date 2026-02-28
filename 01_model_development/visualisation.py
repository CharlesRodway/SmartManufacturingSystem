# Visualisation Script - Per-Machine Bearing Analysis
# Digital Systems Project - Charles Rodway
#
# Generates one 4-panel plot per lathe showing all 4 bearings as separate lines.
# Failing bearings should clearly stand out against healthy ones.
#
# Known failures:
#   Lathe 1 (1st test): Bearing 3 (inner race), Bearing 4 (rolling element)
#   Lathe 2 (2nd test): Bearing 1 (outer race)
#   Lathe 3 (3rd test): Bearing 3 (outer race)
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
        "title": "Lathe 1 — 1st Test Dataset",
        "bearings": ["bearing1", "bearing2", "bearing3", "bearing4"],
        "known_failures": {
            "bearing3": "Inner Race Defect",
            "bearing4": "Rolling Element Defect",
        },
    },
    "lathe_2": {
        "title": "Lathe 2 — 2nd Test Dataset",
        "bearings": ["bearing1", "bearing2", "bearing3", "bearing4"],
        "known_failures": {
            "bearing1": "Outer Race Failure",
        },
    },
    "lathe_3": {
        "title": "Lathe 3 — 3rd Test Dataset",
        "bearings": ["bearing1", "bearing2", "bearing3", "bearing4"],
        "known_failures": {
            "bearing3": "Outer Race Failure",
        },
    },
}

# Consistent colours for each bearing across all plots
BEARING_COLOURS = {
    "bearing1": "#2196F3",  # blue
    "bearing2": "#4CAF50",  # green
    "bearing3": "#F44336",  # red
    "bearing4": "#FF9800",  # orange
}


# ============ PLOT ONE LATHE ============

def plot_lathe(lathe_name, config):
    bearings = config["bearings"]
    known_failures = config["known_failures"]
    cutoff_pct = TRAIN_SPLIT * 100

    # Load all bearing result CSVs
    bearing_data = {}
    for bearing in bearings:
        csv_path = RESULTS_DIR / f"{lathe_name}_{bearing}_results.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found, skipping {bearing}.")
            continue
        df = pd.read_csv(csv_path)
        df['progress'] = (df.index / len(df)) * 100
        bearing_data[bearing] = df

    if not bearing_data:
        print(f"  ERROR: No data found for {lathe_name}. Run train_isolation_forest.py first.")
        return

    print(f"\nGenerating plot for {lathe_name.upper()}...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Build subtitle showing known failures
    failure_str = ", ".join([
        f"Bearing {b[-1]} ({v})"
        for b, v in known_failures.items()
    ])
    fig.suptitle(
        f'Bearing Anomaly Detection — {config["title"]}\n'
        f'Known failures: {failure_str}',
        fontsize=13, fontweight='bold'
    )

    def add_cutoff(ax):
        ax.axvline(x=cutoff_pct, color='black', linestyle='--',
                   linewidth=1.2, alpha=0.6, label='Training cutoff')

    def bearing_label(bearing):
        num = bearing[-1]
        if bearing in known_failures:
            return f"Bearing {num} ⚠ ({known_failures[bearing]})"
        return f"Bearing {num} (healthy)"

    # ---- Plot 1: Anomaly rate per bearing ----
    ax1 = axes[0, 0]
    for bearing, df in bearing_data.items():
        df['rolling_anomaly'] = df['is_anomaly'].rolling(
            window=50, center=True).mean() * 100
        colour = BEARING_COLOURS[bearing]
        lw = 1.5 if bearing in known_failures else 0.8
        alpha = 0.9 if bearing in known_failures else 0.6
        ax1.plot(df['progress'], df['rolling_anomaly'],
                 color=colour, linewidth=lw, alpha=alpha,
                 label=bearing_label(bearing))
    add_cutoff(ax1)
    ax1.set_xlabel('Test Progress (%)')
    ax1.set_ylabel('Anomaly Rate (%)')
    ax1.set_title('Anomaly Rate Over Time')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=5, color='gray', linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Plot 2: Kurtosis per bearing ----
    ax2 = axes[0, 1]
    for bearing, df in bearing_data.items():
        if 'kurtosis' not in df.columns:
            continue
        colour = BEARING_COLOURS[bearing]
        lw = 1.5 if bearing in known_failures else 0.8
        alpha = 0.9 if bearing in known_failures else 0.6
        ax2.plot(df['progress'], df['kurtosis'],
                 color=colour, linewidth=lw, alpha=alpha,
                 label=bearing_label(bearing))
    add_cutoff(ax2)
    ax2.axhline(y=0, color='green', linestyle='--',
                alpha=0.5, label='Normal kurtosis ~0 (Fisher)')
    ax2.set_xlabel('Test Progress (%)')
    ax2.set_ylabel('Kurtosis (Fisher)')
    ax2.set_title('Kurtosis Over Time')
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: Anomaly scores per bearing ----
    ax3 = axes[1, 0]
    for bearing, df in bearing_data.items():
        colour = BEARING_COLOURS[bearing]
        lw = 1.5 if bearing in known_failures else 0.8
        alpha = 0.9 if bearing in known_failures else 0.5
        # Rolling average for cleaner lines
        df['rolling_score'] = df['anomaly_score'].rolling(
            window=30, center=True).mean()
        ax3.plot(df['progress'], df['rolling_score'],
                 color=colour, linewidth=lw, alpha=alpha,
                 label=bearing_label(bearing))
    add_cutoff(ax3)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Test Progress (%)')
    ax3.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax3.set_title('Anomaly Scores Over Time')
    ax3.set_xlim(0, 100)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Plot 4: RMS energy per bearing ----
    ax4 = axes[1, 1]
    for bearing, df in bearing_data.items():
        if 'rms' not in df.columns:
            continue
        colour = BEARING_COLOURS[bearing]
        lw = 1.5 if bearing in known_failures else 0.8
        alpha = 0.9 if bearing in known_failures else 0.6
        ax4.plot(df['progress'], df['rms'],
                 color=colour, linewidth=lw, alpha=alpha,
                 label=bearing_label(bearing))
    add_cutoff(ax4)
    ax4.set_xlabel('Test Progress (%)')
    ax4.set_ylabel('RMS (vibration energy)')
    ax4.set_title('RMS Energy Over Time')
    ax4.set_xlim(0, 100)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / f"{lathe_name}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

    # ---- Terminal summary ----
    print(f"\n  Summary:")
    for bearing, df in bearing_data.items():
        total = len(df)
        anomalies = df['is_anomaly'].sum()
        pct = 100 * anomalies / total
        max_kurt = df['kurtosis'].max() if 'kurtosis' in df.columns else 0
        status = "⚠ FAILING" if bearing in known_failures else "  healthy"
        print(f"  {status} | Bearing {bearing[-1]}: "
              f"{anomalies}/{total} anomalies ({pct:.1f}%) | "
              f"Max kurtosis: {max_kurt:.2f}")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("Visualisation — All Lathes")
    print("=" * 50)

    for lathe_name, config in TEST_CONFIG.items():
        plot_lathe(lathe_name, config)

    print(f"\n{'='*50}")
    print("All plots generated.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
