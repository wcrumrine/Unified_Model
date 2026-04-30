#!/usr/bin/env python3
"""
Halo Data CSV Extractor — CDM + WDM (with/without LSST cuts)
=============================================================
Generates 4 CSV files of per-subhalo halo + galaxy properties for
downstream GH-connection work, using the exact Census pipeline
infrastructure as mcut_scatter_raw_vs_cut.py.

Files produced (default seed=42, default WDM=8 keV):
  halo_data_CDM_no_cuts_seed42.csv
  halo_data_CDM_lsst_cuts_seed42.csv
  halo_data_WDM_8keV_no_cuts_seed42.csv
  halo_data_WDM_8keV_lsst_cuts_seed42.csv

Halo properties extracted (per subhalo):
  host_id                  -- which host halo (e.g. 23)
  R_vir_host_kpc           -- host R_vir, kpc
  r_gal_kpc                -- 3D galactocentric distance, kpc
  mpeak                    -- peak virial mass, M_sun  (Halo_subs['mpeak'])
  vpeak                    -- peak velocity, km/s       (Halo_subs['vpeak'])
  r_acc_kpc                -- subhalo R_vir at accretion, kpc
                             (= halo_total['rvir'] / H_PARAM)

Galaxy properties (one Census stochastic realization, fixed seed):
  M_V, r_half_kpc, mu_V, M_star

Per-cosmology weights:
  base_survival            -- baryonic disruption × f_gal (cosmo-independent)
  f_wdm                    -- WDM suppression factor (= 1 for CDM)
  weight                   -- base_survival × f_wdm  (the w_i in the LaTeX)

Selection logic:
  'no_cuts'   = subhalos with finite M_V (matches 'raw' in mcut_scatter_*)
  'lsst_cuts' = no_cuts subset, additionally surviving:
                  M_V < 0   AND   mu_V < 32 mag/arcsec^2
                (the 10 pc size cut is intentionally NOT applied)

Usage:
  python -u extract_halo_data_csv.py
  python -u extract_halo_data_csv.py --halos 23 88 --wdm_keV 8 --seed 42
  python -u extract_halo_data_csv.py --halos 23           # halo-23 only

Run from a directory at the same depth as mcut_scatter_raw_vs_cut.py
(i.e. CARC_Cluster/Analyses/<some_dir>/), so that ../../utils/data_loader.py
and ../../wdm/ resolve correctly.
"""
print("Cell Initiated")

import os
import sys

# Match mcut_scatter_raw_vs_cut.py: go up TWO levels to reach CARC_Cluster/,
# then add CARC_Cluster/utils and CARC_Cluster/wdm to sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CARC_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_CARC_ROOT, 'utils'))
sys.path.insert(0, os.path.join(_CARC_ROOT, 'wdm'))

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import erf
from numpy.random import default_rng

import data_loader
import model

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION  (mirrors mcut_scatter_raw_vs_cut.py exactly)
# =============================================================================
H_PARAM = 0.7
CDM_MHM = 5.0   # log10(M_hm) for CDM — far below all subhalo masses, ~no suppression

# Nadler (2024) Fig. 14 best-fit GH connection parameters (unchanged)
BASE_PARAMS = {
    'alpha': -1.453, 'sigma_M': 0.14, 'mpeak_cut': 5.94, 'B': 0.98,
    'sigma_mpeak': 0.2, 'A': 0.038, 'sigma_r': 0.71, 'n': 0.75,
    'gamma_M': 0.0, 'xi_8': 0.0, 'xi_9': 0.0, 'xi_10': 0.0,
}


def mWDM_to_log10_Mhm(m_WDM_keV):
    """Half-mode mass from WDM particle mass (Nadler et al. 2021)."""
    M_hm = 5e8 * (m_WDM_keV / 3.0) ** (-10. / 3.)
    return np.log10(M_hm)


# =============================================================================
# IDENTICAL HELPERS — copied verbatim from mcut_scatter_raw_vs_cut.py
# =============================================================================
def stellar_mass_from_MV(M_V, B_V=0.66):
    M_V_sun = 4.81
    a_V = -0.628
    b_V = 1.305
    logML = a_V + b_V * np.array(B_V)
    logL = -0.4 * (np.array(M_V) - M_V_sun)
    logM = logL + logML
    logM -= 0.10
    return 10 ** logM


def compute_mu_V(M_V, r12_kpc):
    """V-band central surface brightness (mag/arcsec^2). Matches the
    formula inside apply_lsst_cuts in mcut_scatter_raw_vs_cut.py."""
    return M_V + 36.57 + 2.5 * np.log10(2.0 * np.pi * r12_kpc ** 2)


def apply_lsst_cuts_no_10pc(M_V, r12_kpc):
    """LSST cuts WITHOUT the 10 pc size cut — only M_V<0 and mu_V<32."""
    mu_V = compute_mu_V(M_V, r12_kpc)
    return (M_V < 0.0) & (mu_V < 32.0)


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
        r_gal_kpc = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * 1000.0 / H_PARAM
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
    f_gal[mask_8] *= 10. ** params['xi_8']
    f_gal[mask_9] *= 10. ** params['xi_9']
    f_gal[mask_10] *= 10. ** params['xi_10']
    baryonic_survival = 1.0 - (halo_data['Halo_ML_prob']) ** (1.0 / params['B'])
    return baryonic_survival * f_gal


def compute_wdm_factor(mpeak, log10_Mhm, h=0.7):
    M_hm = 10. ** log10_Mhm
    return (1.0 + (2.7 * h * M_hm / mpeak)) ** (-0.99)


def generate_galaxy_properties(halo_data, params, seed=42, _cached_interp=[None]):
    """One stochastic realization of (M_V, r_1/2) from the Census model."""
    model.rng = default_rng(seed)
    if _cached_interp[0] is None:
        _cached_interp[0] = data_loader.load_interpolator()
    vpeak_Mr_interp = _cached_interp[0]
    M_V, r12, _ = model.properties_given_theta_multiple(
        params['alpha'], halo_data['Halo_subs'], halo_data['rvir'],
        params['B'], halo_data['Halo_ML_prob'],
        params['sigma_M'], params['gamma_M'], params['sigma_r'],
        params['sigma_mpeak'], params['A'], params['n'], CDM_MHM,
        1, params['mpeak_cut'],
        params['xi_8'], params['xi_9'], params['xi_10'],
        vpeak_Mr_interp,
    )
    M_V = M_V[0, :]
    r12 = r12[0, :]
    valid = np.isfinite(M_V)
    return M_V, r12, valid


# =============================================================================
# EXTRACTION
# =============================================================================
def build_subhalo_table(halos, seed):
    """
    Build a DataFrame of all 'valid' (finite-M_V) subhalos with their halo
    + galaxy properties. No cosmology weighting yet — that's per-CSV.
    """
    halo_data_all = data_loader.load_halo_data_all()
    halo_data_list = data_loader.load_halo_data(halos, halo_data_all)
    host_props = get_host_properties(halo_data_list, halos)
    for hid, props in host_props.items():
        print(f"  Host {hid}: R_vir = {props['rvir_kpc']:.1f} kpc, "
              f"M_vir = {props['mvir']:.3e} M_sun")

    # Per-subhalo arrays (concatenated across hosts in the same order
    # as halo_numbers — load_halo_data_total uses the same order).
    r_gal_kpc, host_ids, R_vir_host = compute_galactocentric_distances(
        halo_data_list, halos, host_props
    )
    halo_total = data_loader.load_halo_data_total(halo_data_list, halos)
    mpeak  = halo_total['Halo_subs']['mpeak']
    vpeak  = halo_total['Halo_subs']['vpeak']   # confirmed: same access used in model.properties_given_theta(_multiple)
    r_acc_kpc = halo_total['rvir'] / H_PARAM   # rvir at accretion, kpc/h → kpc

    n_total = len(mpeak)
    print(f"  Total subhalos (across hosts): {n_total}")

    print(f"  Generating one Census galaxy realization (seed={seed})...")
    base_surv = compute_base_survival(halo_total, BASE_PARAMS)
    M_V_full, r12_full, valid = generate_galaxy_properties(
        halo_total, BASE_PARAMS, seed=seed
    )

    idx_all = np.arange(n_total)
    df = pd.DataFrame({
        'halo_subs_idx':   idx_all[valid],
        'host_id':         host_ids[valid],
        'R_vir_host_kpc':  R_vir_host[valid],
        'r_gal_kpc':       r_gal_kpc[valid],
        'mpeak':           mpeak[valid],
        'vpeak':           vpeak[valid],
        'r_acc_kpc':       r_acc_kpc[valid],
        'M_V':             M_V_full[valid],
        'r_half_kpc':      r12_full[valid],
        'mu_V':            compute_mu_V(M_V_full[valid], r12_full[valid]),
        'M_star':          stellar_mass_from_MV(M_V_full[valid]),
        'base_survival':   base_surv[valid],
    })
    print(f"  Subhalos with finite M_V (= 'no_cuts' row count): {len(df)}")
    # Return df + the per-row mpeak (used for cosmology-dependent f_wdm)
    return df, mpeak[valid]


def write_csv(df, mpeak_valid, cosmo_name, log10_Mhm, apply_cuts,
              out_dir, seed):
    """Add per-cosmology weights, apply cuts if requested, write CSV."""
    f_wdm = compute_wdm_factor(mpeak_valid, log10_Mhm)
    df_out = df.copy()
    df_out['f_wdm']  = f_wdm
    df_out['weight'] = df_out['base_survival'].values * f_wdm
    df_out['cosmo']  = cosmo_name

    if apply_cuts:
        mask = apply_lsst_cuts_no_10pc(
            df_out['M_V'].values, df_out['r_half_kpc'].values
        )
        df_out = df_out[mask].reset_index(drop=True)
        cut_label = 'lsst_cuts'
    else:
        cut_label = 'no_cuts'

    fname = f'halo_data_{cosmo_name}_{cut_label}_seed{seed}.csv'
    fpath = os.path.join(out_dir, fname)
    df_out.to_csv(fpath, index=False)
    n_eff = float(df_out['weight'].sum())
    print(f"    {fname}: {len(df_out):>5d} rows, N_eff = {n_eff:7.1f}")
    return fpath


# =============================================================================
# CLI + MAIN
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Extract per-subhalo CSV (CDM + WDM, with/without LSST cuts).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--halos', type=int, nargs='+', default=[23],
                   help='Host halo IDs (default: 23)')
    p.add_argument('--wdm_keV', type=float, default=8.0,
                   help='WDM particle mass for the WDM CSVs, in keV (default: 8)')
    p.add_argument('--seed', type=int, default=42,
                   help='Census stochastic-realization seed (default: 42)')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory (default: ./Output relative to script)')
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()

    out_dir = args.out_dir or os.path.join(_THIS_DIR, 'Output')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("HALO DATA CSV EXTRACTOR — CDM + WDM, with/without LSST cuts")
    print("=" * 70)
    print(f"  Halos:     {args.halos}")
    print(f"  WDM mass:  {args.wdm_keV} keV")
    print(f"  Seed:      {args.seed}")
    print(f"  Out dir:   {out_dir}")
    print(f"  LSST cuts: M_V < 0  AND  mu_V < 32   (NO 10 pc cut)")

    # Build the per-subhalo table once (same draw for every output CSV).
    print(f"\n{'-' * 70}\nLOADING HALO DATA + EXTRACTING\n{'-' * 70}")
    df, mpeak_valid = build_subhalo_table(args.halos, args.seed)

    # Two cosmologies × two cut variants = 4 CSVs.
    cosmologies = [
        ('CDM',                          CDM_MHM),
        (f'WDM_{int(args.wdm_keV)}keV',  mWDM_to_log10_Mhm(args.wdm_keV)),
    ]
    print(f"\n{'-' * 70}\nWRITING CSVs\n{'-' * 70}")
    written = []
    for cosmo_name, log10_Mhm in cosmologies:
        print(f"  {cosmo_name}  (log10 M_hm = {log10_Mhm:.3f}):")
        for apply_cuts in [False, True]:
            fpath = write_csv(df, mpeak_valid, cosmo_name, log10_Mhm,
                              apply_cuts, out_dir, args.seed)
            written.append(fpath)

    print(f"\n{'=' * 70}")
    print(f"DONE — {(time.time() - t0) / 60:.2f} min")
    print(f"Files in: {out_dir}")
    for f in written:
        print(f"  {os.path.basename(f)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

print("Cell Completed")
