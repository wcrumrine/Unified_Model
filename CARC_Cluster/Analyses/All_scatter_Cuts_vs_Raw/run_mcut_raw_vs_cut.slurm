#!/bin/bash -l
#SBATCH --partition=gluscevic
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -J Mcut_RawVsCut
#SBATCH --output=Mcut_RawVsCut_%j.log
#SBATCH --mem=64G

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "Started at: $(date)"
echo "=========================================="

module purge

source /spack/conda/miniconda3/23.3.1/etc/profile.d/conda.sh
conda activate MW_Likelihood

cd ~/Unified_Model/Unified_Model_Analysis/CARC_Cluster/Analyses/All_scatter_Cuts_vs_Raw/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ""
echo "=========================================="
echo ">>> M_cut SCATTER: RAW vs LSST-CUT <<<"
echo "=========================================="
echo ""
echo "  30 stochastic Census realizations × 12 cosmologies"
echo "  Two data types per realization:"
echo "    - 'raw':  all galaxies (no LSST cuts)"
echo "    - 'cut':  LSST fiducial cuts applied"
echo "  Pass 1 only (all 9 shape params free)"
echo "  20 bins × 20 bins"
echo "  GH connection params: fixed at Nadler 2024 best-fit"
echo "  Using $SLURM_CPUS_PER_TASK CPU cores"
echo "  Expected time: ~10-15 min"
echo ""

python -u mcut_scatter_raw_vs_cut.py

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
