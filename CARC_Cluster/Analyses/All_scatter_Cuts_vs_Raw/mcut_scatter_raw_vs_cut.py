#!/usr/bin/env python3
"""
M_cut Scatter Exploration — RAW vs LSST-CUT (Stage 1 v2, CARC)
===============================================================
Question: Is the sigma_size-vs-m_WDM trend a real physical response, or
an LSST selection artifact?

Procedure:
  For each of 30 seeds, extract ONE set of stochastic galaxy properties
  (M_V, r_1/2) from halo 23. Produce TWO data versions:
    - 'raw': all extracted galaxies (no LSST cuts applied)
    - 'cut': LSST fiducial cuts applied (M_V<0, r>10pc, mu_V<32)
  Both use the same underlying stochastic draw, so they differ only by
  the selection. Bin both at 20 bins, fit Pass 1 (all 9 shape params free)
  using scipy optimization.

  If sigma_size trends with m_WDM only in the 'cut' version → selection
  artifact. If it trends in both → physical response.

This script is Pass 1 only (all 9 free). Pass 2 is not needed for the
sigma_size question and is skipped to save compute.

Seeds: same 30 seeds as original exploration (SEED_START=300, 300..329).
This is intentional — shared seeds across cosmologies give a paired
comparison (same galaxies, different physics).

Usage:
  python -u mcut_scatter_raw_vs_cut.py

Run from: /home1/crumrine/Unified_Model/2401.10318/Dwarf_Data_Extraction_10
(requires ../utils/data_loader.py, ../utils/model.py)

Outputs (to OUTPUT_DIR):
  mcut_scatter_raw_vs_cut_20bins.pkl
"""
print("Cell Initiated")

import sys
import os
# utils lives at CARC_Cluster/utils — go up two levels from Analyses/All_scatter_Cuts_vs_Raw/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, '..', '..', 'utils')))

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import warnings
import time
import pickle
import multiprocessing
from multiprocessing import Pool

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

import data_loader
import model
from numpy.random import default_rng

warnings.filterwarnings('ignore')
_np_trapz = getattr(np, 'trapezoid', None) or np.trapz

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_PATH, 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CORES = 16
N_REAL = 30
SEED_START = 300

M_STAR_MIN = 2.0
M_STAR_MAX = 9.0

N_BINS_M_RGAL = 20        # 20-bin only for Stage 1
DATA_TYPES = ['raw', 'cut']

H_PARAM = 0.7

WDM_MASSES_KEV = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
CDM_MHM = 5.0

COSMO_NAMES = ['CDM'] + [f'WDM_{m}keV' for m in WDM_MASSES_KEV]
MWDM_VALUES = [100.0] + list(map(float, WDM_MASSES_KEV))
COSMO_TO_MWDM = dict(zip(COSMO_NAMES, MWDM_VALUES))

# =============================================================================
# NADLER (2024) BEST-FIT GH CONNECTION PARAMETERS
# =============================================================================
BASE_PARAMS = {
    'alpha': -1.453, 'sigma_M': 0.14, 'mpeak_cut': 5.94, 'B': 0.98,
    'sigma_mpeak': 0.2, 'A': 0.038, 'sigma_r': 0.71, 'n': 0.75,
    'gamma_M': 0.0, 'xi_8': 0.0, 'xi_9': 0.0, 'xi_10': 0.0,
}

def mWDM_to_log10_Mhm(m_WDM_keV):
    M_hm = 5e8 * (m_WDM_keV / 3.0)**(-10./3.)
    return np.log10(M_hm)

SCENARIOS = {'CDM': {'m_wdm_keV': 100, 'Mhm': CDM_MHM}}
for m in WDM_MASSES_KEV:
    SCENARIOS[f'WDM_{m}keV'] = {'m_wdm_keV': m, 'Mhm': mWDM_to_log10_Mhm(m)}

# =============================================================================
# FITTING BOUNDS (no power law — M_cut is direct)
# =============================================================================
BOUNDS_9 = [
    (5.0, 800.0),    # A
    (1.5, 6.0),      # M_0
    (-1.5, 0.5),     # alpha
    (0.5, 5.0),      # M_cut
    (0.02, 0.5),     # a_size
    (0.5, 2.5),      # b_size
    (0.1, 1.5),      # sigma_size
    (-1.5, 2.0),     # gamma
    (0.05, 2.0),     # x_s
]

# =============================================================================
# RECONFIGURABLE GRID STATE
# =============================================================================
N_QUAD = 30
_GNFW_NORM_GRID = np.linspace(0.001, 1.0, 2000)

_N_BINS_M = N_BINS_M_RGAL
_N_BINS_RGAL = N_BINS_M_RGAL
_BINS_M = np.linspace(M_STAR_MIN, M_STAR_MAX, _N_BINS_M + 1)
_BINS_RGAL = np.linspace(0, 1, _N_BINS_RGAL + 1)
_RGAL_BIN_CENTERS = 0.5 * (_BINS_RGAL[:-1] + _BINS_RGAL[1:])
_RGAL_BIN_WIDTHS = _BINS_RGAL[1:] - _BINS_RGAL[:-1]

def _build_smf_quadrature(bins_M):
    n_bins = len(bins_M) - 1
    quad_points = np.zeros((n_bins, N_QUAD))
    quad_weights = np.zeros((n_bins, N_QUAD))
    for k in range(n_bins):
        pts = np.linspace(bins_M[k], bins_M[k + 1], N_QUAD)
        quad_points[k] = pts
        dx = pts[1] - pts[0]
        w = np.full(N_QUAD, dx)
        w[0] = dx / 2
        w[-1] = dx / 2
        quad_weights[k] = w
    return quad_points, quad_weights

_SMF_QUAD_POINTS, _SMF_QUAD_WEIGHTS = _build_smf_quadrature(_BINS_M)
_SMF_QUAD_FLAT = _SMF_QUAD_POINTS.ravel()

# =============================================================================
# PARAMETRIC MODELS
# =============================================================================
def single_schechter_with_cutoff(log_mstar, A, M_0, alpha, M_cut):
    mstar = 10.0 ** log_mstar
    m0 = 10.0 ** M_0
    mcut = 10.0 ** M_cut
    return A * (mstar / m0) ** alpha * np.exp(-mcut / mstar)

def _vectorized_smf_model_counts(A, M_0, alpha, M_cut):
    smf_vals = single_schechter_with_cutoff(_SMF_QUAD_FLAT, A, M_0, alpha, M_cut)
    smf_2d = smf_vals.reshape(_N_BINS_M, N_QUAD)
    return np.sum(smf_2d * _SMF_QUAD_WEIGHTS, axis=1)

def _vectorized_poisson_deviance(model_arr, observed):
    model_arr = np.maximum(model_arr, 1e-10)
    dev = np.sum(model_arr)
    mask = observed > 0
    if np.any(mask):
        dev += np.sum(-observed[mask] + observed[mask] * np.log(observed[mask] / model_arr[mask]))
    return dev

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================
def stellar_mass_from_MV(M_V, B_V=0.66):
    M_V_sun = 4.81
    a_V = -0.628
    b_V = 1.305
    logML = a_V + b_V * np.array(B_V)
    logL = -0.4 * (np.array(M_V) - M_V_sun)
    logM = logL + logML
    logM -= 0.10
    return 10**logM

def apply_lsst_cuts(M_V, r12_kpc):
    r12_pc = r12_kpc * 1000.0
    mu_V = M_V + 36.57 + 2.5 * np.log10(2.0 * np.pi * r12_kpc**2)
    return (M_V < 0.0) & (r12_pc > 10.0) & (mu_V < 32.0)

def get_host_properties(halo_data_list, halo_numbers):
    host_props = {}
    for i, halo_id in enumerate(halo_numbers):
        host = halo_data_list[i]['Halo_main'][0]
        host_props[halo_id] = {
            'x': host['x'], 'y': host['y'], 'z': host['z'],
            'rvir_kpc': host['rvir'] / H_PARAM,
            'mvir': host['mvir'],
        }
    return host_props

def compute_galactocentric_distances(halo_data_list, halo_numbers, host_props):
    r_gal_list, host_id_list, rvir_host_list = [], [], []
    for i, halo_id in enumerate(halo_numbers):
        Halo_subs = halo_data_list[i]['Halo_subs']
        n_subs = len(Halo_subs['x'])
        xc = host_props[halo_id]['x']
        yc = host_props[halo_id]['y']
        zc = host_props[halo_id]['z']
        dx = Halo_subs['x'] - xc
        dy = Halo_subs['y'] - yc
        dz = Halo_subs['z'] - zc
        r_gal_kpc = np.sqrt(dx**2 + dy**2 + dz**2) * 1000.0 / H_PARAM
        r_gal_list.append(r_gal_kpc)
        host_id_list.append(np.full(n_subs, halo_id, dtype=int))
        rvir_host_list.append(np.full(n_subs, host_props[halo_id]['rvir_kpc']))
    return (np.concatenate(r_gal_list),
            np.concatenate(host_id_list),
            np.concatenate(rvir_host_list))

def compute_base_survival(halo_data, params, h=0.7):
    mpeak = halo_data['Halo_subs']['mpeak']
    log_mpeak = np.log10(mpeak)
    log_h = np.log10(h)
    f_gal = 0.5 * (1.0 + erf(
        (log_mpeak - params['mpeak_cut']) / (np.sqrt(2) * params['sigma_mpeak'])
    ))
    mask_8 = np.logical_and(log_mpeak - log_h > 7.5, log_mpeak - log_h < 8.5)
    mask_9 = np.logical_and(log_mpeak - log_h > 8.5, log_mpeak - log_h < 9.5)
    mask_10 = np.logical_and(log_mpeak - log_h > 9.5, log_mpeak - log_h < 10.5)
    f_gal[mask_8] *= 10.**params['xi_8']
    f_gal[mask_9] *= 10.**params['xi_9']
    f_gal[mask_10] *= 10.**params['xi_10']
    baryonic_survival = 1.0 - (halo_data['Halo_ML_prob'])**(1.0 / params['B'])
    return baryonic_survival * f_gal

def compute_wdm_factor(mpeak, log10_Mhm, h=0.7):
    M_hm = 10.**log10_Mhm
    return (1.0 + (2.7 * h * M_hm / mpeak))**(-0.99)

def generate_galaxy_properties(halo_data, params, seed=42, _cached_interp=[None]):
    """Generate one realization of M_V and r_1/2 from the Census model."""
    model.rng = default_rng(seed)
    alpha = params['alpha']
    sigma_M = params['sigma_M']
    sigma_r = params['sigma_r']
    sigma_mpeak = params['sigma_mpeak']
    if _cached_interp[0] is None:
        _cached_interp[0] = data_loader.load_interpolator()
    vpeak_Mr_interp = _cached_interp[0]
    M_V, r12, _ = model.properties_given_theta_multiple(
        alpha, halo_data['Halo_subs'], halo_data['rvir'],
        params['B'], halo_data['Halo_ML_prob'],
        sigma_M, params['gamma_M'], sigma_r, sigma_mpeak,
        params['A'], params['n'], CDM_MHM,
        1, params['mpeak_cut'],
        params['xi_8'], params['xi_9'], params['xi_10'],
        vpeak_Mr_interp
    )
    M_V = M_V[0, :]
    r12 = r12[0, :]
    valid = np.isfinite(M_V)
    return M_V, r12, valid


def extract_raw_and_cut(halo_total, r_gal, host_ids, R_vir_hosts,
                        base_surv, seed):
    """
    Generate ONE stochastic realization, then produce TWO versions:
      'raw': all galaxies with finite M_V (no LSST cuts)
      'cut': galaxies surviving LSST cuts (M_V<0, r>10pc, mu_V<32)

    Both share the same underlying draw. Returns nested dict:
      {data_type: {cosmo_name: {log_mstar, log_rhalf, x_rgal, weights, N_eff}}}
    """
    MV, r12, valid = generate_galaxy_properties(halo_total, BASE_PARAMS, seed=seed)
    mpeak = halo_total['Halo_subs']['mpeak']

    # Step 1: finite-M_V mask (valid extraction)
    MV_v = MV[valid]
    r12_v = r12[valid]
    mpeak_v = mpeak[valid]
    rgal_v = r_gal[valid]
    Rvir_host_v = R_vir_hosts[valid]
    base_v = base_surv[valid]

    mstar_v = stellar_mass_from_MV(MV_v)
    log_mstar_v = np.log10(mstar_v)
    log_rhalf_v = np.log10(r12_v * 1000.0)  # kpc → pc
    x_rgal_v = rgal_v / Rvir_host_v

    # Step 2: LSST mask (applied for 'cut' only)
    lsst_mask = apply_lsst_cuts(MV_v, r12_v)

    # --- Assemble per-cosmology data for each data_type ---
    per_type = {}
    for data_type in DATA_TYPES:
        sel = slice(None) if data_type == 'raw' else np.where(lsst_mask)[0]
        per_cosmo = {}
        for cosmo_name, sp in SCENARIOS.items():
            f_wdm = compute_wdm_factor(mpeak_v, sp['Mhm'])
            weights_full = base_v * f_wdm
            per_cosmo[cosmo_name] = {
                'log_mstar': log_mstar_v[sel],
                'log_rhalf': log_rhalf_v[sel],
                'x_rgal':    x_rgal_v[sel],
                'weights':   weights_full[sel],
                'N_eff':     np.sum(weights_full[sel]),
            }
        per_type[data_type] = per_cosmo
    return per_type


def bin_realization_data(raw_per_cosmo):
    """Bin per-cosmology data using current grid settings."""
    binned = {}
    for cosmo_name, d in raw_per_cosmo.items():
        counts_M, _ = np.histogram(d['log_mstar'], bins=_BINS_M, weights=d['weights'])
        counts_rgal, _ = np.histogram(d['x_rgal'], bins=_BINS_RGAL, weights=d['weights'])
        binned[cosmo_name] = {
            'log_mstar': d['log_mstar'],
            'log_rhalf': d['log_rhalf'],
            'x_rgal': d['x_rgal'],
            'weights': d['weights'],
            'N_eff': d['N_eff'],
            'counts_M': counts_M,
            'counts_rgal': counts_rgal,
        }
    return binned

# =============================================================================
# LIKELIHOOD (weighted, all 9 free)
# =============================================================================
def neg_log_likelihood_single(theta9, data):
    A, M_0, alpha, M_cut, a_size, b_size, sigma_size, gamma, x_s = theta9
    if sigma_size <= 0 or A <= 0:
        return 1e20

    N_eff = data['N_eff']
    w = data['weights']

    # SMF
    model_counts = _vectorized_smf_model_counts(A, M_0, alpha, M_cut)
    nll = _vectorized_poisson_deviance(model_counts, data['counts_M'])

    # Size-mass (per galaxy, weighted)
    inv_sigma2 = 1.0 / (sigma_size * sigma_size)
    log_sigma = np.log(sigma_size)
    residuals = data['log_rhalf'] - (a_size * data['log_mstar'] + b_size)
    nll += 0.5 * np.sum(w * residuals * residuals) * inv_sigma2 + N_eff * log_sigma

    # Radial
    gnfw_norm_integrand = _GNFW_NORM_GRID ** (2 - gamma) * \
                          (1 + _GNFW_NORM_GRID / x_s) ** (gamma - 3)
    gnfw_norm = _np_trapz(gnfw_norm_integrand, _GNFW_NORM_GRID)
    gnfw_shape = _RGAL_BIN_CENTERS ** (2 - gamma) * \
                 (1 + _RGAL_BIN_CENTERS / x_s) ** (gamma - 3)
    gnfw_pdf = gnfw_shape / gnfw_norm if gnfw_norm > 0 else np.ones(_N_BINS_RGAL)
    model_rgal = gnfw_pdf * _RGAL_BIN_WIDTHS * N_eff
    nll += _vectorized_poisson_deviance(model_rgal, data['counts_rgal'])

    return nll

def fit_all_free(data, cosmo_name):
    try:
        result_de = differential_evolution(
            neg_log_likelihood_single, BOUNDS_9, args=(data,),
            seed=12345, maxiter=300, tol=1e-6, polish=False,
            workers=1, updating='immediate',
        )
        result_bfgs = minimize(
            neg_log_likelihood_single, result_de.x, args=(data,),
            method='L-BFGS-B', bounds=BOUNDS_9,
            options={'maxiter': 500, 'ftol': 1e-10},
        )
        if result_bfgs.fun < result_de.fun and np.all(np.isfinite(result_bfgs.x)):
            return result_bfgs.x, result_bfgs.fun
        else:
            return result_de.x, result_de.fun
    except Exception as e:
        print(f"    WARNING: fit_all_free failed for {cosmo_name}: {e}")
        return np.full(9, np.nan), np.nan

def _fit_allfree_job(args):
    real_idx, cosmo_name, data_type, data = args
    theta9, nll = fit_all_free(data, cosmo_name)
    return {
        'real_idx': real_idx,
        'cosmo': cosmo_name,
        'data_type': data_type,
        'mcut_allfree': theta9[3] if not np.isnan(theta9[3]) else np.nan,
        'theta9_allfree': theta9,
        'nll_allfree': nll,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total_start = time.time()
    print("=" * 70)
    print("M_cut SCATTER — RAW vs LSST-CUT (Stage 1 v2)")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cores: {N_CORES}")
    print(f"Realizations: {N_REAL}")
    print(f"Cosmologies: {len(COSMO_NAMES)}")
    print(f"Host halo: 23 only")
    print(f"Data types: {DATA_TYPES}")
    print(f"Bin config: {N_BINS_M_RGAL} × {N_BINS_M_RGAL}")
    print(f"Pass: all 9 params free only (Pass 1)")

    # ------------------------------------------------------------------
    # Load halo data
    # ------------------------------------------------------------------
    print(f"\n{'='*70}\nLOADING HALO DATA (halo 23 only)\n{'='*70}")
    halo_numbers = [23]
    halo_data_all = data_loader.load_halo_data_all()
    halo_data_list = data_loader.load_halo_data(halo_numbers, halo_data_all)
    host_props = get_host_properties(halo_data_list, halo_numbers)
    for hid, props in host_props.items():
        print(f"  Halo {hid}: R_vir = {props['rvir_kpc']:.1f} kpc")
    r_gal, host_ids, R_vir_hosts = compute_galactocentric_distances(
        halo_data_list, halo_numbers, host_props
    )
    halo_total = data_loader.load_halo_data_total(halo_data_list, halo_numbers)
    n_subhalos = len(halo_total['Halo_subs']['mpeak'])
    print(f"  Total subhalos: {n_subhalos}")
    base_surv = compute_base_survival(halo_total, BASE_PARAMS)

    # ------------------------------------------------------------------
    # Extract all 30 realizations (both raw + cut at once)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"EXTRACTING {N_REAL} STOCHASTIC REALIZATIONS (raw + cut)")
    print(f"{'='*70}")

    all_data = []   # list of {data_type: {cosmo: {...}}}
    t0 = time.time()
    for real_idx in range(N_REAL):
        seed = SEED_START + real_idx
        per_type = extract_raw_and_cut(
            halo_total, r_gal, host_ids, R_vir_hosts, base_surv, seed
        )
        all_data.append(per_type)

        if (real_idx + 1) % 5 == 0 or real_idx == 0:
            n_raw = len(per_type['raw']['CDM']['log_mstar'])
            n_cut = len(per_type['cut']['CDM']['log_mstar'])
            print(f"  Realization {real_idx+1:>3}/{N_REAL}: seed={seed}, "
                  f"N_raw(CDM)={n_raw}, N_cut(CDM)={n_cut}")
    t_extract = (time.time() - t0) / 60
    print(f"\n  Extraction complete: {t_extract:.1f} min")

    # ------------------------------------------------------------------
    # Bin and fit — loop over data_types
    # ------------------------------------------------------------------
    all_results = []
    for data_type in DATA_TYPES:
        label = f"{data_type}_{N_BINS_M_RGAL}bins"
        print(f"\n{'#'*70}\n# FITTING: data_type={data_type}, bins={N_BINS_M_RGAL}\n{'#'*70}")

        # Bin
        print(f"\n  Binning {N_REAL} realizations ({data_type})...")
        all_binned = [bin_realization_data(all_data[i][data_type])
                      for i in range(N_REAL)]

        # Build job list
        jobs = []
        for real_idx in range(N_REAL):
            for cosmo_name in COSMO_NAMES:
                d = all_binned[real_idx][cosmo_name]
                jobs.append((real_idx, cosmo_name, data_type, d))
        print(f"  Total jobs: {len(jobs)}")

        # Fit
        print(f"  Started: {time.strftime('%H:%M:%S')}")
        t0 = time.time()
        if N_CORES > 1:
            with Pool(N_CORES) as pool:
                results = pool.map(_fit_allfree_job, jobs)
        else:
            results = [_fit_allfree_job(j) for j in jobs]
        t_fit = (time.time() - t0) / 60
        print(f"  Complete: {t_fit:.1f} min [{time.strftime('%H:%M:%S')}]")

        all_results.extend(results)

        # Quick per-cosmo summary for this data_type
        df_dt = pd.DataFrame(results)
        print(f"\n  {data_type} summary (mean ± σ across {N_REAL} realizations):")
        print(f"  {'Cosmology':<12} {'m_WDM':>6} {'M_cut':>10} {'σ':>8} "
              f"{'sigma_size':>12} {'σ':>8}")
        print(f"  {'-'*60}")
        for cosmo in COSMO_NAMES:
            m_wdm = COSMO_TO_MWDM[cosmo]
            m_label = 'CDM' if m_wdm > 50 else f'{m_wdm:.0f}'
            sub = df_dt[df_dt['cosmo'] == cosmo]
            mcut_vals = sub['mcut_allfree'].dropna()
            sigsize_vals = np.array([
                row['theta9_allfree'][6]
                for _, row in sub.iterrows()
                if hasattr(row['theta9_allfree'], '__len__')
                and len(row['theta9_allfree']) == 9
                and np.isfinite(row['theta9_allfree'][6])
            ])
            print(f"  {cosmo:<12} {m_label:>6} "
                  f"{mcut_vals.mean():>10.4f} {mcut_vals.std():>8.4f} "
                  f"{sigsize_vals.mean():>12.4f} {sigsize_vals.std():>8.4f}")

    # ------------------------------------------------------------------
    # Save combined pkl
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    pkl_path = os.path.join(OUTPUT_DIR,
                            f'mcut_scatter_raw_vs_cut_{N_BINS_M_RGAL}bins.pkl')
    save_data = {
        'results_df': results_df,
        'config': {
            'N_REAL': N_REAL, 'SEED_START': SEED_START,
            'N_BINS_M': N_BINS_M_RGAL, 'N_BINS_RGAL': N_BINS_M_RGAL,
            'M_STAR_MIN': M_STAR_MIN, 'M_STAR_MAX': M_STAR_MAX,
            'DATA_TYPES': DATA_TYPES,
            'GH_PARAMS': 'Nadler_2024_bestfit',
            'HOST_HALOS': [23],
            'PASS': 'all_9_free_only',
        },
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n  Saved: {pkl_path}")

    t_total = (time.time() - t_total_start) / 60
    print(f"\n{'='*70}")
    print(f"DONE — {t_total:.1f} min total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

print("Cell Completed")
