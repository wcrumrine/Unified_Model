"""Read Dwarf_SC_Scatter_Data.csv and visualize the cluster and dwarf samples
as a scatter plot on log-log axes.

Usage:
    python plot_scatter.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path(__file__).parent / "Dwarf_SC_Scatter_Data.csv"
OUT_PATH = Path(__file__).parent / "scatter_plot.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows from {CSV_PATH.name}")
    print(df.describe())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df["x_cluster"],
        df["y_cluster"],
        s=8,
        alpha=0.5,
        color="tab:blue",
        label="clusters",
        edgecolors="none",
    )
    ax.scatter(
        df["x_dwarfs"],
        df["y_dwarfs"],
        s=8,
        alpha=0.5,
        color="tab:orange",
        label="dwarfs",
        edgecolors="none",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dwarf / Cluster scatter data")
    ax.legend(frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved figure to {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
