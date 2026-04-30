"""Draw 50 independent resamples from the cluster-points gaussian_kde fit
and plot them on a single log-log axis, color-coded by resample index.

Each resample has the same size as the original cluster sample so that any
single color is directly comparable to the original scatter.

Usage:
    python plot_kde_many_resamples.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

CSV_PATH = Path(__file__).parent / "Dwarf_SC_Scatter_Data.csv"
OUT_PATH = Path(__file__).parent / "kde_resamples_50.png"

N_RESAMPLES = 50
RNG = np.random.default_rng(0)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    clusters = df[["x_cluster", "y_cluster"]].dropna()
    n = len(clusters)
    print(f"Fitting KDE to {n:,} cluster points (in log10 space)")

    log_data = np.log10(clusters.to_numpy()).T
    kde = gaussian_kde(log_data)

    cmap = plt.get_cmap("turbo", N_RESAMPLES)
    norm = Normalize(vmin=0, vmax=N_RESAMPLES - 1)

    fig, ax = plt.subplots(figsize=(9, 7))

    for i in range(N_RESAMPLES):
        log_sample = kde.resample(size=n, seed=RNG)
        ax.scatter(
            10 ** log_sample[0],
            10 ** log_sample[1],
            s=5,
            color=cmap(i),
            alpha=0.35,
            edgecolors="none",
        )

    ax.scatter(
        clusters["x_cluster"], clusters["y_cluster"],
        s=10, facecolors="none", edgecolors="black", linewidths=0.6,
        label=f"original cluster points ({n})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x_cluster")
    ax.set_ylabel("y_cluster")
    ax.set_title(
        f"{N_RESAMPLES} independent gaussian_kde.resample() draws "
        f"(size={n} each), color-coded by draw index"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0, N_RESAMPLES - 1, 6))
    cbar.set_label("resample index")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved figure to {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
