#!/usr/bin/env bash
#SBATCH --job-name=jaxley-hoc-export
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Run this once on a CPU node where NEURON and the required MOD mechanisms are
# installed. GPU jobs can then load the JSON/NPZ artifact without NEURON.
set -euo pipefail

CONFIG=${1:?"usage: sbatch export_hoc_artifact.sbatch CONFIG DESTINATION"}
DESTINATION=${2:?"usage: sbatch export_hoc_artifact.sbatch CONFIG DESTINATION"}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=${MPLBACKEND:-Agg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${SLURM_TMPDIR:-/tmp}/matplotlib"}
export NEURON_MODULE_OPTIONS=${NEURON_MODULE_OPTIONS:--nogui}
mkdir -p logs "${MPLCONFIGDIR}"

srun "${PYTHON_EXECUTABLE}" -m jaxley_refactored.cli.bootstrap \
  export-hoc-artifact --config "${CONFIG}" --destination "${DESTINATION}"
