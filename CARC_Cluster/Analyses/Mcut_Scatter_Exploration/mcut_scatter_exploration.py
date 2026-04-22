#!/usr/bin/env python3
"""
M_cut Scatter Exploration — Extract + Fit (CARC)
==================================================
Question: How much does M_cut scatter when fitting to Census Pipeline data,
accounting for random galaxy formation scatter?

Procedure:
  For each m_WDM, generate N_REAL stochastic mock realizations from the
  Census Pipeline (holding GH connection params fixed at Nadler 2024 best-fit;
  only luminosity scatter and size scatter vary across realizations), fit the
  SGN model to each using scipy optimization, then record M_cut vs. WDM mass.

Two fitting versions (run in two passes per binning config):
  Pass 1 — All 9 shape params free (M_cut, A, M_0, alpha, a_size, b_size,
     sigma_size, gamma, x_s). Reveals how parameter degeneracies (especially
     A–M_cut) absorb realization-to-realization fluctuations.
  Pass 2 — Only M_cut free, other 8 shape params fixed to the MEDIAN of the
     Pass 1 all-free results (across all realizations and cosmologies).
     Isolates stochastic variance in M_cut with no room for other params to
     compensate. Pass 2 depends on Pass 1 completing first.

Two binning configurations (run sequentially on the same extracted data):
  - 20 stellar mass bins × 20 radial bins (current pipeline default)
  -  9 stellar mass bins ×  9 radial bins (fewer, wider bins to reduce
     noise from sparse bins with single-host galaxy counts)

GH connection params: fixed at Nadler (2024, LSST Forecasts Fig. 14) best-fit.

Single host halo (ID 23) only — LSST observes one Milky Way.

Extraction uses LSST fiducial detection cuts (M_V<0, r_1/2>10pc, mu_V<32)
to be consistent with Stage 1 calibration.

Stochastic scatter sources (2 total, both from model.py):
  - Luminosity scatter (sigma_M = 0.14): lognormal on Vpeak → M_V
  - Size scatter (sigma_r = 0.71): lognormal on r_1/2
  The occupation fraction (f_gal) is deterministic given fixed GH params.

Usage:
  python -u mcut_scatter_exploration.py

Run from: /home1/crumrine/Unified_Model/2401.10318/Dwarf_Data_Extraction_10
(requires ../utils/data_loader.py, ../utils/model.py)

Outputs (to OUTPUT_DIR, one set per binning config):
  mcut_scatter_results_{N}bins.pkl
  mcut_scatter_allfree_{N}bins.png
  mcut_scatter_mcutonly_{N}bins.png
  mcut_scatter_combined_{N}bins.png
  mcut_scatter_A_vs_Mcut_{N}bins.png
"""
print("Cell Initiated")

import sys
sys.path.append('../utils')

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time
import pickle
import os
import multiprocessing
from multiprocessing import Pool

# Force fork on Linux — ensures child processes inherit module-level globals
# set by reinit_grids(). 'spawn' would re-import the module and reset to defaults.
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set; fine on CARC

import data_loader
import model
from numpy.random import default_rng

warnings.filterwarnings('ignore')

# NumPy compatibility
_np_trapz = getattr(np, 'trapezoid', None) or np.trapz

# =============================================================================
# GLOBAL PLOT STYLE
# =============================================================================
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 26,
})

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = '/home1/crumrine/Unified_Model/2401.10318/Proof_of_Concept_2/'
OUTPUT_DIR = BASE_PATH + 'mcut_scatter_exploration/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CORES = 16
N_REAL = 30
SEED_START = 300

# Stellar mass range (fixed across all binning configs)
M_STAR_MIN = 2.0
M_STAR_MAX = 9.0

# Binning configs to compare
BIN_CONFIGS = [20, 9]

H_PARAM = 0.7

# Cosmologies
WDM_MASSES_KEV = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
CDM_MHM = 5.0

COSMO_NAMES = ['CDM'] + [f'WDM_{m}keV' for m in WDM_MASSES_KEV]
MWDM_VALUES = [100.0] + list(map(float, WDM_MASSES_KEV))
COSMO_TO_MWDM = dict(zip(COSMO_NAMES, MWDM_VALUES))

M_WDM_PIVOT = 10.0
LOG10_PIVOT = np.log10(M_WDM_PIVOT)

# =============================================================================
# NADLER (2024) BEST-FIT GH CONNECTION PARAMETERS
# =============================================================================
BASE_PARAMS = {
    'alpha': -1.453,
    'sigma_M': 0.14,
    'mpeak_cut': 5.94,
    'B': 0.98,
    'sigma_mpeak': 0.2,
    'A': 0.038,
    'sigma_r': 0.71,
    'n': 0.75,
    'gamma_M': 0.0,
    'xi_8': 0.0,
    'xi_9': 0.0,
    'xi_10': 0.0,
}

# =============================================================================
# WDM SCENARIOS
# =============================================================================
def mWDM_to_log10_Mhm(m_WDM_keV):
    M_hm = 5e8 * (m_WDM_keV / 3.0)**(-10./3.)
    return np.log10(M_hm)

SCENARIOS = {'CDM': {'m_wdm_keV': 100, 'Mhm': CDM_MHM}}
for m in WDM_MASSES_KEV:
    SCENARIOS[f'WDM_{m}keV'] = {'m_wdm_keV': m, 'Mhm': mWDM_to_log10_Mhm(m)}

# Stage 1 reference (for truth lines only, NOT used as fixed params)
S1_REFERENCE = {
    'A': 222.6, 'M_0': 2.296, 'alpha': -0.347,
    'M_cut_10': 1.897, 'b_pow': 2.114,
    'a_size': 0.192, 'b_size': 0.978,
    'sigma_size': 0.366, 'gamma': -0.991, 'x_s': 0.198,
}

# =============================================================================
# FITTING BOUNDS (no power law — M_cut is a direct param)
# =============================================================================
BOUNDS_9 = [
    (5.0, 800.0),       # A
    (1.5, 6.0),          # M_0
    (-1.5, 0.5),         # alpha
    (0.5, 5.0),          # M_cut
    (0.02, 0.5),         # a_size
    (0.5, 2.5),          # b_size
    (0.1, 1.5),          # sigma_size
    (-1.5, 2.0),         # gamma
    (0.05, 2.0),         # x_s
]

BOUNDS_MCUT_ONLY = [(0.5, 5.0)]

# =============================================================================
# RECONFIGURABLE GRID STATE
# =============================================================================
# These module-level variables get reassigned by reinit_grids() before each
# fitting pipeline. Worker processes (Pool.map) inherit the values set at
# the time the Pool is created.

N_QUAD = 30
_GNFW_NORM_GRID = np.linspace(0.001, 1.0, 2000)

# Initialized to defaults; overwritten before fitting
_N_BINS_M = 20
_N_BINS_RGAL = 20
_BINS_M = np.linspace(M_STAR_MIN, M_STAR_MAX, _N_BINS_M + 1)
_BINS_RGAL = np.linspace(0, 1, _N_BINS_RGAL + 1)
_RGAL_BIN_CENTERS = 0.5 * (_BINS_RGAL[:-1] + _BINS_RGAL[1:])
_RGAL_BIN_WIDTHS = _BINS_RGAL[1:] - _BINS_RGAL[:-1]
_SMF_QUAD_POINTS = None
_SMF_QUAD_WEIGHTS = None
_SMF_QUAD_FLAT = None


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


def reinit_grids(n_bins_m, n_bins_rgal, verbose=True):
    """Reassign module-level grid variables for a new binning config."""
    global _N_BINS_M, _N_BINS_RGAL
    global _BINS_M, _BINS_RGAL
    global _RGAL_BIN_CENTERS, _RGAL_BIN_WIDTHS
    global _SMF_QUAD_POINTS, _SMF_QUAD_WEIGHTS, _SMF_QUAD_FLAT

    _N_BINS_M = n_bins_m
    _N_BINS_RGAL = n_bins_rgal
    _BINS_M = np.linspace(M_STAR_MIN, M_STAR_MAX, n_bins_m + 1)
    _BINS_RGAL = np.linspace(0, 1, n_bins_rgal + 1)
    _RGAL_BIN_CENTERS = 0.5 * (_BINS_RGAL[:-1] + _BINS_RGAL[1:])
    _RGAL_BIN_WIDTHS = _BINS_RGAL[1:] - _BINS_RGAL[:-1]
    _SMF_QUAD_POINTS, _SMF_QUAD_WEIGHTS = _build_smf_quadrature(_BINS_M)
    _SMF_QUAD_FLAT = _SMF_QUAD_POINTS.ravel()

    if verbose:
        print(f"  Grids initialized: {n_bins_m} M* bins × {n_bins_rgal} r_gal bins")
        print(f"  M* bin width: {_BINS_M[1]-_BINS_M[0]:.3f} dex, "
              f"r_gal bin width: {_BINS_RGAL[1]-_BINS_RGAL[0]:.3f}")


# Initialize once at import time (silent)
reinit_grids(20, 20, verbose=False)


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


def _vectorized_poisson_deviance(model, observed):
    model = np.maximum(model, 1e-10)
    dev = np.sum(model)
    mask = observed > 0
    if np.any(mask):
        dev += np.sum(-observed[mask] + observed[mask] * np.log(observed[mask] / model[mask]))
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


def generate_galaxy_properties(halo_data, params, seed=42, nonstochastic=False,
                               _cached_interp=[None]):
    """Generate ONE realization of galaxy properties (M_V, r_1/2).

    Two stochastic draws (both from model.py):
      - Luminosity: lognormal scatter (sigma_M) on Vpeak → M_V
      - Size: lognormal scatter (sigma_r) on r_1/2

    NOTE on seeding: model.py uses np.random.lognormal (legacy API), so
    setting model.rng = default_rng(seed) may NOT control reproducibility.
    Different seeds still produce different draws within a single run
    (numpy global state advances), but results may not be reproducible
    across separate runs. Same behavior as v3/v4 extraction scripts.
    Check if model.py on CARC has been updated to use rng.lognormal.
    """
    model.rng = default_rng(seed)
    alpha = params['alpha']
    sigma_M = 0.0 if nonstochastic else params['sigma_M']
    sigma_r = 0.0 if nonstochastic else params['sigma_r']
    sigma_mpeak = 0.001 if nonstochastic else params['sigma_mpeak']
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


def extract_single_realization(halo_total, r_gal, host_ids, R_vir_hosts,
                               base_surv, seed):
    """
    Generate one stochastic realization and return per-cosmology RAW data
    (no binning — binning happens later so we can re-bin with different configs).

    Order of operations:
      1. Stochastic draw of M_V and r_1/2 from Census model
         (luminosity scatter + size scatter vary per seed)
      2. LSST detection cuts on the surviving galaxies
      3. For each cosmology: multiply base survival by WDM suppression
         to get final per-galaxy weights

    Returns:
        dict: cosmo_name -> {log_mstar, log_rhalf, x_rgal, weights, N_eff}
    """
    MV, r12, valid = generate_galaxy_properties(halo_total, BASE_PARAMS, seed=seed)
    mpeak = halo_total['Halo_subs']['mpeak']

    # Step 1: mask out halos with non-finite M_V
    MV_v = MV[valid]
    r12_v = r12[valid]
    mpeak_v = mpeak[valid]
    rgal_v = r_gal[valid]
    Rvir_host_v = R_vir_hosts[valid]
    base_v = base_surv[valid]

    # Step 2: LSST cuts
    lsst_mask = apply_lsst_cuts(MV_v, r12_v)
    MV_v = MV_v[lsst_mask]
    r12_v = r12_v[lsst_mask]
    mpeak_v = mpeak_v[lsst_mask]
    rgal_v = rgal_v[lsst_mask]
    Rvir_host_v = Rvir_host_v[lsst_mask]
    base_v = base_v[lsst_mask]

    mstar_v = stellar_mass_from_MV(MV_v)
    log_mstar_v = np.log10(mstar_v)
    log_rhalf_v = np.log10(r12_v * 1000.0)  # kpc → pc
    x_rgal_v = rgal_v / Rvir_host_v

    # Step 3: per-cosmology weights (no histograms yet)
    per_cosmo = {}
    for cosmo_name, sp in SCENARIOS.items():
        f_wdm = compute_wdm_factor(mpeak_v, sp['Mhm'])
        weights = base_v * f_wdm
        N_eff = np.sum(weights)

        per_cosmo[cosmo_name] = {
            'log_mstar': log_mstar_v,
            'log_rhalf': log_rhalf_v,
            'x_rgal': x_rgal_v,
            'weights': weights,
            'N_eff': N_eff,
        }
    return per_cosmo


def bin_realization_data(raw_per_cosmo):
    """
    Bin raw per-cosmology data using the CURRENT module-level grid settings.
    Call reinit_grids() before this to change binning.
    """
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
# PER-COSMOLOGY WEIGHTED LIKELIHOOD
# =============================================================================

def neg_log_likelihood_single(theta9, data):
    """Weighted NLL for a single cosmology. 9 free params."""
    A, M_0, alpha, M_cut, a_size, b_size, sigma_size, gamma, x_s = theta9

    if sigma_size <= 0 or A <= 0:
        return 1e20

    N_eff = data['N_eff']
    w = data['weights']

    # SMF
    model_counts = _vectorized_smf_model_counts(A, M_0, alpha, M_cut)
    nll = _vectorized_poisson_deviance(model_counts, data['counts_M'])

    # Size-mass
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


def neg_log_likelihood_mcut_only(theta1, data, fixed_params):
    """Weighted NLL with only M_cut free."""
    M_cut = theta1[0]
    A = fixed_params['A']
    M_0 = fixed_params['M_0']
    alpha = fixed_params['alpha']
    a_size = fixed_params['a_size']
    b_size = fixed_params['b_size']
    sigma_size = fixed_params['sigma_size']
    gamma = fixed_params['gamma']
    x_s = fixed_params['x_s']

    N_eff = data['N_eff']
    w = data['weights']

    model_counts = _vectorized_smf_model_counts(A, M_0, alpha, M_cut)
    nll = _vectorized_poisson_deviance(model_counts, data['counts_M'])

    inv_sigma2 = 1.0 / (sigma_size * sigma_size)
    log_sigma = np.log(sigma_size)
    residuals = data['log_rhalf'] - (a_size * data['log_mstar'] + b_size)
    nll += 0.5 * np.sum(w * residuals * residuals) * inv_sigma2 + N_eff * log_sigma

    gnfw_norm_integrand = _GNFW_NORM_GRID ** (2 - gamma) * \
                          (1 + _GNFW_NORM_GRID / x_s) ** (gamma - 3)
    gnfw_norm = _np_trapz(gnfw_norm_integrand, _GNFW_NORM_GRID)
    gnfw_shape = _RGAL_BIN_CENTERS ** (2 - gamma) * \
                 (1 + _RGAL_BIN_CENTERS / x_s) ** (gamma - 3)
    gnfw_pdf = gnfw_shape / gnfw_norm if gnfw_norm > 0 else np.ones(_N_BINS_RGAL)
    model_rgal = gnfw_pdf * _RGAL_BIN_WIDTHS * N_eff
    nll += _vectorized_poisson_deviance(model_rgal, data['counts_rgal'])

    return nll


# =============================================================================
# SCIPY FITTING WRAPPERS
# =============================================================================

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


def fit_mcut_only(data, cosmo_name, fixed_params):
    try:
        result_de = differential_evolution(
            neg_log_likelihood_mcut_only, BOUNDS_MCUT_ONLY,
            args=(data, fixed_params),
            seed=12345, maxiter=200, tol=1e-7, polish=True,
            workers=1, updating='immediate',
        )
        return np.array([result_de.x[0]]), result_de.fun
    except Exception as e:
        print(f"    WARNING: fit_mcut_only failed for {cosmo_name}: {e}")
        return np.array([np.nan]), np.nan


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================

def _fit_allfree_job(args):
    real_idx, cosmo_name, data = args
    theta9, nll = fit_all_free(data, cosmo_name)
    return {
        'real_idx': real_idx,
        'cosmo': cosmo_name,
        'mcut_allfree': theta9[3] if not np.isnan(theta9[3]) else np.nan,
        'theta9_allfree': theta9,
        'nll_allfree': nll,
    }


def _fit_mcutonly_job(args):
    real_idx, cosmo_name, data, fixed_params = args
    theta1, nll = fit_mcut_only(data, cosmo_name, fixed_params)
    return {
        'real_idx': real_idx,
        'cosmo': cosmo_name,
        'mcut_mcutonly': theta1[0],
        'nll_mcutonly': nll,
    }


# =============================================================================
# HELPER
# =============================================================================

def s1_mcut_for_cosmo(m_wdm):
    return S1_REFERENCE['M_cut_10'] + S1_REFERENCE['b_pow'] * (LOG10_PIVOT - np.log10(m_wdm))


# =============================================================================
# PLOTTING
# =============================================================================

def plot_mcut_scatter(results_df, col, version_label, n_bins, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    x_plot, y_mean, y_std, y_all = [], [], [], []

    for cosmo in COSMO_NAMES:
        m_wdm = COSMO_TO_MWDM[cosmo]
        if m_wdm > 15:
            continue
        sub = results_df[results_df['cosmo'] == cosmo]
        vals = sub[col].dropna().values
        if len(vals) == 0:
            continue
        x_plot.append(m_wdm)
        y_mean.append(np.mean(vals))
        y_std.append(np.std(vals))
        y_all.append(vals)

    x_plot = np.array(x_plot)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)

    ax.fill_between(x_plot, y_mean - y_std, y_mean + y_std,
                    alpha=0.25, color='#2E86AB', label=r'$\pm 1\sigma$ scatter')

    for i, m_wdm in enumerate(x_plot):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(y_all[i]))
        ax.scatter(np.full_like(y_all[i], m_wdm) + jitter,
                   y_all[i], s=12, alpha=0.3, color='#2E86AB', zorder=2)

    ax.plot(x_plot, y_mean, 'o-', color='#2E86AB', ms=10, lw=2.5,
            label=f'Mean ({version_label})', zorder=4)

    mcut_truth = np.array([s1_mcut_for_cosmo(m) for m in x_plot])
    ax.plot(x_plot, mcut_truth, 's--', color='#C44536', ms=10, lw=2,
            label='Stage 1 power-law truth', zorder=5)

    ax.set_xlabel(r'$m_{\mathrm{WDM}}$ [keV]')
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{cut}} / M_\odot)$')
    ax.set_title(f'$M_{{\\mathrm{{cut}}}}$ Scatter — {version_label}\n'
                 f'({N_REAL} realizations, halo 23, {n_bins} bins, GH fixed)')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined(results_df, n_bins, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'allfree': '#2E86AB', 'mcutonly': '#D4733C'}
    labels = {'allfree': 'All 9 params free', 'mcutonly': r'$M_{\rm cut}$ only free'}
    offsets = {'allfree': -0.15, 'mcutonly': 0.15}

    for version in ['allfree', 'mcutonly']:
        col = f'mcut_{version}'
        x_plot, y_mean, y_std = [], [], []

        for cosmo in COSMO_NAMES:
            m_wdm = COSMO_TO_MWDM[cosmo]
            if m_wdm > 15:
                continue
            sub = results_df[results_df['cosmo'] == cosmo]
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            x_plot.append(m_wdm)
            y_mean.append(np.mean(vals))
            y_std.append(np.std(vals))

        x_plot = np.array(x_plot)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)

        ax.errorbar(x_plot + offsets[version], y_mean, yerr=y_std,
                    fmt='o-', capsize=4, capthick=2, ms=8,
                    color=colors[version], ecolor=colors[version], lw=2,
                    label=f'{labels[version]} (mean ± 1σ)', zorder=4)

    m_wdm_grid = np.array([m for m in MWDM_VALUES if m <= 15])
    mcut_truth = np.array([s1_mcut_for_cosmo(m) for m in m_wdm_grid])
    ax.plot(m_wdm_grid, mcut_truth, 's--', color='#C44536', ms=10, lw=2,
            label='Stage 1 power-law truth', zorder=5)

    ax.set_xlabel(r'$m_{\mathrm{WDM}}$ [keV]')
    ax.set_ylabel(r'$\log_{10}(M_{\mathrm{cut}} / M_\odot)$')
    ax.set_title(f'$M_{{\\mathrm{{cut}}}}$ Scatter: All Free vs. Fixed Shape\n'
                 f'({N_REAL} realizations, halo 23, {n_bins} bins, GH fixed)')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_A_vs_Mcut(results_df, n_bins, output_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=5, vmax=15)

    for cosmo in COSMO_NAMES:
        m_wdm = COSMO_TO_MWDM[cosmo]
        if m_wdm > 15:
            continue
        sub = results_df[results_df['cosmo'] == cosmo]
        A_vals, Mcut_vals = [], []
        for _, row in sub.iterrows():
            t9 = row['theta9_allfree']
            if hasattr(t9, '__len__') and len(t9) == 9 and np.isfinite(t9[0]):
                A_vals.append(t9[0])
                Mcut_vals.append(t9[3])
        if len(A_vals) == 0:
            continue
        ax.scatter(Mcut_vals, A_vals,
                   c=[m_wdm]*len(A_vals), cmap=cmap, norm=norm,
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$m_{\mathrm{WDM}}$ [keV]', fontsize=16)

    ax.set_xlabel(r'$\log_{10}(M_{\mathrm{cut}} / M_\odot)$')
    ax.set_ylabel(r'$A$ (SMF amplitude)')
    ax.set_title(f'$A$ vs. $M_{{\\mathrm{{cut}}}}$ — All Free Fits ({n_bins} bins)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FITTING PIPELINE (called once per binning config)
# =============================================================================

def run_fitting_pipeline(all_raw_data, n_bins_m, n_bins_rgal):
    """
    Bin data, run Pass 1 (all free) and Pass 2 (M_cut only), generate plots.

    Args:
        all_raw_data: list of per-cosmo dicts from extract_single_realization()
        n_bins_m: number of stellar mass bins
        n_bins_rgal: number of radial bins

    Returns:
        results_df: merged DataFrame with both pass results
    """
    label = f"{n_bins_m}bins"

    print(f"\n{'#'*70}")
    print(f"# FITTING PIPELINE: {n_bins_m} M* bins × {n_bins_rgal} r_gal bins")
    print(f"{'#'*70}")

    # Reconfigure grids
    reinit_grids(n_bins_m, n_bins_rgal)

    # Bin all realizations with current grid
    print(f"\n  Binning {N_REAL} realizations...")
    all_binned = []
    for real_idx in range(N_REAL):
        binned = bin_realization_data(all_raw_data[real_idx])
        all_binned.append(binned)

    # ------------------------------------------------------------------
    # Pass 1: All 9 params free
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PASS 1 ({label}): ALL 9 PARAMS FREE")
    print(f"{'='*70}")
    print(f"  Started at: {time.strftime('%H:%M:%S')}")

    jobs_p1 = []
    for real_idx in range(N_REAL):
        for cosmo_name in COSMO_NAMES:
            data = all_binned[real_idx][cosmo_name]
            jobs_p1.append((real_idx, cosmo_name, data))

    print(f"  Total jobs: {len(jobs_p1)}")

    t0 = time.time()
    if N_CORES > 1:
        with Pool(N_CORES) as pool:
            results_p1 = pool.map(_fit_allfree_job, jobs_p1)
    else:
        results_p1 = [_fit_allfree_job(j) for j in jobs_p1]

    t_p1 = (time.time() - t0) / 60
    print(f"\n  Pass 1 complete: {t_p1:.1f} min [{time.strftime('%H:%M:%S')}]")

    # ------------------------------------------------------------------
    # Compute fixed params for Pass 2 from Pass 1 medians
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"COMPUTING FIXED PARAMS FROM PASS 1 MEDIANS ({label})")
    print(f"{'='*70}")

    all_theta9 = []
    for r in results_p1:
        t9 = r['theta9_allfree']
        if hasattr(t9, '__len__') and len(t9) == 9 and np.all(np.isfinite(t9)):
            all_theta9.append(t9)

    all_theta9 = np.array(all_theta9)
    n_valid = len(all_theta9)
    n_total = len(results_p1)
    print(f"  Valid all-free fits: {n_valid} / {n_total}")
    if n_valid < n_total:
        print(f"  ({n_total - n_valid} fits returned NaN — optimizer exceptions)")

    if n_valid == 0:
        print("  FATAL: All optimizers failed. Cannot compute Pass 2 fixed params.")
        print("  Skipping Pass 2 for this binning config.")
        df_p1 = pd.DataFrame(results_p1)
        return df_p1, t_p1, 0.0

    medians_p1 = np.median(all_theta9, axis=0)

    fixed_params_from_p1 = {
        'A':          medians_p1[0],
        'M_0':        medians_p1[1],
        'alpha':      medians_p1[2],
        'a_size':     medians_p1[4],
        'b_size':     medians_p1[5],
        'sigma_size': medians_p1[6],
        'gamma':      medians_p1[7],
        'x_s':        medians_p1[8],
    }

    print(f"\n  Fixed params for Pass 2 (median of {n_valid} all-free fits):")
    for k, v in fixed_params_from_p1.items():
        print(f"    {k:<12} = {v:.4f}")

    # ------------------------------------------------------------------
    # Pass 2: M_cut only free
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PASS 2 ({label}): M_cut ONLY FREE")
    print(f"{'='*70}")
    print(f"  Started at: {time.strftime('%H:%M:%S')}")

    jobs_p2 = []
    for real_idx in range(N_REAL):
        for cosmo_name in COSMO_NAMES:
            data = all_binned[real_idx][cosmo_name]
            jobs_p2.append((real_idx, cosmo_name, data, fixed_params_from_p1))

    print(f"  Total jobs: {len(jobs_p2)}")

    t0 = time.time()
    if N_CORES > 1:
        with Pool(N_CORES) as pool:
            results_p2 = pool.map(_fit_mcutonly_job, jobs_p2)
    else:
        results_p2 = [_fit_mcutonly_job(j) for j in jobs_p2]

    t_p2 = (time.time() - t0) / 60
    print(f"\n  Pass 2 complete: {t_p2:.1f} min [{time.strftime('%H:%M:%S')}]")

    # ------------------------------------------------------------------
    # Merge and summarize
    # ------------------------------------------------------------------
    df_p1 = pd.DataFrame(results_p1)
    df_p2 = pd.DataFrame(results_p2)
    results_df = df_p1.merge(df_p2, on=['real_idx', 'cosmo'], how='outer')

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY ({label})")
    print(f"{'='*70}")

    print(f"\n  {'Cosmology':<12} {'m_WDM':>6} "
          f"{'M_cut(all)':>12} {'σ(all)':>8} "
          f"{'M_cut(fix)':>12} {'σ(fix)':>8} "
          f"{'S1 truth':>10}")
    print(f"  {'-'*74}")

    for cosmo in COSMO_NAMES:
        m_wdm = COSMO_TO_MWDM[cosmo]
        sub = results_df[results_df['cosmo'] == cosmo]
        mcut_all = sub['mcut_allfree'].dropna()
        mcut_fix = sub['mcut_mcutonly'].dropna()
        mcut_s1 = s1_mcut_for_cosmo(m_wdm)
        m_label = 'CDM' if m_wdm > 50 else f'{m_wdm:.0f}'

        print(f"  {cosmo:<12} {m_label:>6} "
              f"{mcut_all.mean():>12.4f} {mcut_all.std():>8.4f} "
              f"{mcut_fix.mean():>12.4f} {mcut_fix.std():>8.4f} "
              f"{mcut_s1:>10.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pkl_path = os.path.join(OUTPUT_DIR, f'mcut_scatter_results_{label}.pkl')
    save_data = {
        'results_df': results_df,
        'fixed_params_from_p1': fixed_params_from_p1,
        's1_reference': S1_REFERENCE,
        'config': {
            'N_REAL': N_REAL, 'SEED_START': SEED_START,
            'N_BINS_M': n_bins_m, 'N_BINS_RGAL': n_bins_rgal,
            'M_STAR_MIN': M_STAR_MIN, 'M_STAR_MAX': M_STAR_MAX,
            'LSST_CUTS': True, 'GH_PARAMS': 'Nadler_2024_bestfit',
            'HOST_HALOS': [23],
        },
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n  Saved: {pkl_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"GENERATING PLOTS ({label})")
    print(f"{'='*70}")

    plot_mcut_scatter(results_df, 'mcut_allfree', 'All 9 params free', n_bins_m,
                      os.path.join(OUTPUT_DIR, f'mcut_scatter_allfree_{label}.png'))

    plot_mcut_scatter(results_df, 'mcut_mcutonly', r'$M_{\rm cut}$ only free', n_bins_m,
                      os.path.join(OUTPUT_DIR, f'mcut_scatter_mcutonly_{label}.png'))

    plot_combined(results_df, n_bins_m,
                  os.path.join(OUTPUT_DIR, f'mcut_scatter_combined_{label}.png'))

    plot_A_vs_Mcut(results_df, n_bins_m,
                   os.path.join(OUTPUT_DIR, f'mcut_scatter_A_vs_Mcut_{label}.png'))

    return results_df, t_p1, t_p2


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_total_start = time.time()

    print("=" * 70)
    print("M_cut SCATTER EXPLORATION")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cores: {N_CORES}")
    print(f"Realizations: {N_REAL}")
    print(f"Cosmologies: {len(COSMO_NAMES)}")
    print(f"Host halo: 23 only (single MW)")
    print(f"Binning configs: {BIN_CONFIGS}")

    # ------------------------------------------------------------------
    # 1. Load halo data — SINGLE HOST (halo 23)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LOADING HALO DATA (halo 23 only)")
    print(f"{'='*70}")

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

    # Sanity checks for single-host data
    assert n_subhalos > 0, "No subhalos loaded — check halo data path"
    assert n_subhalos == len(r_gal), \
        f"Index mismatch: {n_subhalos} subhalos vs {len(r_gal)} r_gal entries"
    assert 'Halo_ML_prob' in halo_total, "Missing Halo_ML_prob in halo_total"
    print(f"  Single-host sanity checks passed ✓")

    base_surv = compute_base_survival(halo_total, BASE_PARAMS)
    print(f"  Base survival: mean={base_surv.mean():.4f}")

    # ------------------------------------------------------------------
    # 2. Extract all realizations ONCE (no binning yet)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"EXTRACTING {N_REAL} STOCHASTIC REALIZATIONS (LSST cuts applied)")
    print(f"{'='*70}")

    all_raw_data = []
    t0 = time.time()

    for real_idx in range(N_REAL):
        seed = SEED_START + real_idx
        raw_per_cosmo = extract_single_realization(
            halo_total, r_gal, host_ids, R_vir_hosts, base_surv, seed
        )
        all_raw_data.append(raw_per_cosmo)

        if (real_idx + 1) % 5 == 0 or real_idx == 0:
            cdm_neff = raw_per_cosmo['CDM']['N_eff']
            wdm5_neff = raw_per_cosmo['WDM_5keV']['N_eff']
            print(f"  Realization {real_idx+1:>3}/{N_REAL}: seed={seed}, "
                  f"N_eff(CDM)={cdm_neff:.1f}, N_eff(5keV)={wdm5_neff:.1f}")

    t_extract = (time.time() - t0) / 60
    print(f"\n  Extraction complete: {t_extract:.1f} min")

    # ------------------------------------------------------------------
    # 3. Run fitting pipeline for each binning config
    # ------------------------------------------------------------------
    all_pipeline_results = {}

    for n_bins in BIN_CONFIGS:
        results_df, t_p1, t_p2 = run_fitting_pipeline(
            all_raw_data, n_bins, n_bins
        )
        all_pipeline_results[n_bins] = {
            'results_df': results_df,
            't_p1': t_p1, 't_p2': t_p2,
        }

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    t_total = (time.time() - t_total_start) / 3600
    print(f"\n{'='*70}")
    print(f"M_cut SCATTER EXPLORATION COMPLETE — {t_total:.2f} HOURS")
    print(f"{'='*70}")
    print(f"  Extraction: {t_extract:.1f} min")
    for n_bins, info in all_pipeline_results.items():
        print(f"  {n_bins}-bin pipeline: Pass 1 = {info['t_p1']:.1f} min, "
              f"Pass 2 = {info['t_p2']:.1f} min")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

print("Cell Completed")
