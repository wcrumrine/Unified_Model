#!/bin/bash -l
#SBATCH --partition=gluscevic
#SBATCH -t 10:00:00
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -J Mcut_Scatter
#SBATCH --output=Mcut_Scatter_%j.log
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

cd /home1/crumrine/Unified_Model/2401.10318/Mcut_Scatter_Exploration/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ""
echo "=========================================="
echo ">>> M_cut SCATTER EXPLORATION <<<"
echo "=========================================="
echo ""
echo "  30 stochastic Census realizations × 12 cosmologies"
echo "  Two fits per (realization, cosmology):"
echo "    A) All 9 shape params free (DE + L-BFGS-B)"
echo "    B) Only M_cut free, other 8 fixed to Pass 1 medians"
echo "  GH connection params: fixed at Nadler 2024 best-fit"
echo "  LSST cuts applied"
echo "  Using $SLURM_CPUS_PER_TASK CPU cores"
echo "  Expected time: ~1-3 hours"
echo ""

python -u mcut_scatter_exploration.py

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
