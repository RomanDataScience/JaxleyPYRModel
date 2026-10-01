#!/usr/bin/env bash
#SBATCH --job-name=ca1-nrnivmodl
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Compile the MOD files once into .cache/mod/<arch>/ so the array tasks do not
# race each other running nrnivmodl in the same folder.
set -euo pipefail
source "${CA1_SLURM_DIR:-$(dirname -- "$0")}/common.sh"

"${PYTHON_EXECUTABLE}" -c "import ca1_model; print(ca1_model.build_mechanisms(force=True))"
