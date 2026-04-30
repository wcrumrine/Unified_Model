"""Fit a 2D Gaussian KDE to the cluster points in Dwarf_SC_Scatter_Data.csv,
visualize the resulting density, and draw a resample from it.

Because the cluster sample spans several orders of magnitude in both x and y,
the KDE is fit in log10 space (otherwise the bandwidth is dominated by a few
extreme points and the density estimate is essentially invisible).

Usage:
    python plot_kde.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

CSV_PATH = Path(__file__).parent / "Dwarf_SC_Scatter_Data.csv"
DENSITY_OUT = Path(__file__).parent / "kde_density.png"
RESAMPLE_OUT = Path(__file__).parent / "kde_resample.png"

RNG = np.random.default_rng(0)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    clusters = df[["x_cluster", "y_cluster"]].dropna()
    print(f"Using {len(clusters):,} cluster points for the KDE fit")

    # Fit in log10 space; gaussian_kde expects shape (n_dims, n_points).
    log_data = np.log10(clusters.to_numpy()).T
    kde = gaussian_kde(log_data)
    print(f"KDE bandwidth factor (Scott's rule): {kde.factor:.4f}")

    # ---- Plot 1: density on a regular grid in log-log space ----
    pad = 0.3
    log_x = log_data[0]
    log_y = log_data[1]
    x_grid = np.linspace(log_x.min() - pad, log_x.max() + pad, 200)
    y_grid = np.linspace(log_y.min() - pad, log_y.max() + pad, 200)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([XX.ravel(), YY.ravel()])
    ZZ = kde(grid_points).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(
        10**XX, 10**YY, ZZ, cmap="viridis", shading="auto"
    )
    ax.contour(
        10**XX, 10**YY, ZZ,
        levels=6, colors="white", linewidths=0.6, alpha=0.7,
    )
    ax.scatter(
        clusters["x_cluster"], clusters["y_cluster"],
        s=6, color="white", edgecolors="black", linewidths=0.2,
        alpha=0.7, label="cluster points",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x_cluster")
    ax.set_ylabel("y_cluster")
    ax.set_title("gaussian_kde density of cluster points (fit in log space)")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("density in log10(x), log10(y)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(DENSITY_OUT, dpi=150)
    print(f"Saved density figure to {DENSITY_OUT}")

    # ---- Plot 2: resample from the KDE ----
    n_resample = len(clusters)
    log_samples = kde.resample(size=n_resample, seed=RNG)
    samples = 10**log_samples

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(
        clusters["x_cluster"], clusters["y_cluster"],
        s=12, color="tab:blue", alpha=0.6, edgecolors="none",
        label=f"original ({len(clusters)})",
    )
    ax2.scatter(
        samples[0], samples[1],
        s=12, color="tab:red", alpha=0.6, edgecolors="none",
        marker="x", label=f"KDE resample ({n_resample})",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("x_cluster")
    ax2.set_ylabel("y_cluster")
    ax2.set_title("Cluster points vs. gaussian_kde.resample()")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(frameon=False)
    fig2.tight_layout()
    fig2.savefig(RESAMPLE_OUT, dpi=150)
    print(f"Saved resample figure to {RESAMPLE_OUT}")

    plt.show()


if __name__ == "__main__":
    main()
