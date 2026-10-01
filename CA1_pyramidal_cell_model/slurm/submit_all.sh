#!/usr/bin/env bash
# Submit the full CA1 degeneracy study:
#   1. compile the MOD files once,
#   2. a 10-seed CMA-ES array per cell (after 1),
#   3. a landscape/UMAP analysis per cell once its array has ended (after 2).
#
#   CONDA_ENV=ca1 bash CA1_pyramidal_cell_model/slurm/submit_all.sh
#
# Extra sbatch options (partition, account, ...) go in SBATCH_EXTRA, e.g.
#   SBATCH_EXTRA="--partition=cpu --account=mylab" bash .../submit_all.sh
# GENERATIONS and POPULATION_SIZE are passed through (defaults 15 and 30).
set -euo pipefail

SLURM_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CA1_DIR=$(cd -- "${SLURM_DIR}/.." && pwd)
CONFIGS=(configs/cma_m20240527cd.yaml configs/cma_m20260331b.yaml)

cd "${CA1_DIR}"
mkdir -p logs
# The sbatch scripts are copied to a spool directory, so tell them where
# common.sh lives.
export CA1_SLURM_DIR="${SLURM_DIR}"
read -r -a EXTRA <<< "${SBATCH_EXTRA:-}"

build=$(sbatch --parsable ${EXTRA[@]+"${EXTRA[@]}"} --export=ALL "${SLURM_DIR}/build_mechanisms.sbatch")
echo "build mechanisms: job ${build}"

for config in "${CONFIGS[@]}"; do
  cell=$(basename "${config}" .yaml)            # cma_<cell>
  array=$(sbatch --parsable ${EXTRA[@]+"${EXTRA[@]}"} --dependency="afterok:${build}" \
    --job-name="${cell}-seeds" --export="ALL,CONFIG=${config}" \
    "${SLURM_DIR}/cma_seeds.sbatch")
  analysis=$(sbatch --parsable ${EXTRA[@]+"${EXTRA[@]}"} --dependency="afterany:${array}" \
    --job-name="${cell}-landscape" --export="ALL,RUN_PREFIX=runs/${cell}" \
    "${SLURM_DIR}/analyze_landscape.sbatch")
  echo "${cell}: seed array ${array}, analysis ${analysis}"
done
echo "Logs: ${CA1_DIR}/logs; runs: ${CA1_DIR}/runs/<config>_seed<N>; plots: ${CA1_DIR}/plots/landscape_<cell>"
