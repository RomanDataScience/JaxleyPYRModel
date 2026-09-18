#!/usr/bin/env bash
set -euo pipefail

# Create or update one reusable Conda environment for the Jaxley fitting
# pipeline. The script is intentionally non-interactive for batch/HPC use.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_CMD="${CONDA_CMD:-conda}"
ENV_NAME="${ENV_NAME:-jaxley-mocma}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_TESTS="${INSTALL_TESTS:-0}"

if ! command -v "${CONDA_CMD}" >/dev/null 2>&1; then
  echo "Could not find Conda command: ${CONDA_CMD}" >&2
  echo "Load your Conda module or set CONDA_CMD before running this script." >&2
  exit 1
fi

environment_exists="$(${CONDA_CMD} env list | awk -v name="${ENV_NAME}" '$1 == name {print $1}')"
if [[ "${environment_exists}" != "${ENV_NAME}" ]]; then
  echo "Creating Conda environment ${ENV_NAME} with Python ${PYTHON_VERSION}"
  "${CONDA_CMD}" create --yes --channel conda-forge --name "${ENV_NAME}" \
    "python=${PYTHON_VERSION}" pip
else
  echo "Using existing Conda environment ${ENV_NAME}"
fi

CONDA_RUN=("${CONDA_CMD}" run --no-capture-output --name "${ENV_NAME}")

echo "Installing numerical, Jaxley, Optuna, and plotting dependencies"
"${CONDA_RUN[@]}" python -m pip install --upgrade \
  "jax" \
  "jaxley>=0.13" \
  "numpy" \
  "PyYAML" \
  "matplotlib" \
  "optuna>=4.0" \
  "cmaes" \
  "optunahub"

echo "Installing the local Jaxley and feature-fitting packages in editable mode"
"${CONDA_RUN[@]}" python -m pip install --no-deps --editable "${REPO_DIR}/Jaxley_refactored"
"${CONDA_RUN[@]}" python -m pip install --no-deps --editable "${SCRIPT_DIR}"

if [[ "${INSTALL_TESTS}" == "1" ]]; then
  "${CONDA_RUN[@]}" python -m pip install --upgrade pytest
fi

echo "Checking installed packages"
"${CONDA_RUN[@]}" python -m pip check
"${CONDA_RUN[@]}" python -c \
  'import jax, jaxley, optuna, optunahub; print("environment imports: OK")'

echo "Environment ready: ${ENV_NAME}"
echo "Use it with: conda run --name ${ENV_NAME} bash ${SCRIPT_DIR}/run_mocma.sh run"
