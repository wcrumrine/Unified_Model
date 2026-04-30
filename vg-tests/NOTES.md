# Notes — KDE exploration of `Dwarf_SC_Scatter_Data.csv`

Data: `Dwarf_SC_Scatter_Data.csv` with columns `x_cluster, y_cluster, x_dwarfs, y_dwarfs`
(560 valid cluster rows, 11,682 dwarf rows). The cluster sample spans many orders of
magnitude in both axes, so all KDE work is done in `log10` space.

## 1. Read CSV and visualize as scatter plot
Script: `plot_scatter.py` → `scatter_plot.png`

```python
df = pd.read_csv("Dwarf_SC_Scatter_Data.csv")
ax.scatter(df["x_cluster"], df["y_cluster"], label="clusters")
ax.scatter(df["x_dwarfs"],  df["y_dwarfs"],  label="dwarfs")
ax.set_xscale("log"); ax.set_yscale("log")
```

## 2. Fit `scipy.stats.gaussian_kde` to the cluster points and visualize the density
Script: `plot_kde.py` → `kde_density.png`

```python
clusters = df[["x_cluster", "y_cluster"]].dropna()
log_data = np.log10(clusters.to_numpy()).T   # shape (2, n)
kde = gaussian_kde(log_data)                 # Scott's rule, factor ~ 0.348

XX, YY = np.meshgrid(x_grid, y_grid)         # grid in log10 space
ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
ax.pcolormesh(10**XX, 10**YY, ZZ)            # plotted on log-log axes
```

## 3. Generate a plot from `.resample()` of the fit
Script: `plot_kde.py` → `kde_resample.png`

```python
log_samples = kde.resample(size=len(clusters), seed=RNG)
samples = 10 ** log_samples
ax.scatter(samples[0], samples[1], label="KDE resample")
```

## 4. Do the resample 50 times, all on the same plot, color-coded
Script: `plot_kde_many_resamples.py` → `kde_resamples_50.png`

```python
cmap = plt.get_cmap("turbo", 50)
for i in range(50):
    log_sample = kde.resample(size=len(clusters), seed=RNG)
    ax.scatter(10**log_sample[0], 10**log_sample[1], color=cmap(i), alpha=0.35)
```

## Caveats / next steps
- KDE is fit in `log10(x), log10(y)`; converting `kde.pdf` to a density in linear
  `x, y` requires the Jacobian `1 / (x · y · ln(10)^2)`.
- Bandwidth uses Scott's rule by default; switchable via `bw_method=` to
  `"silverman"` or a scalar/array.
- `np.random.default_rng(0)` is used everywhere for reproducibility.
