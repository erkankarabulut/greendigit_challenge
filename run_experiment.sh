#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

# Usage: sbatch run_experiment.sh <python_script> [optional_args]

# Check if script argument is provided
if [ -z "$1" ]; then
    echo "Error: No Python script specified"
    echo "Usage: sbatch run_experiment.sh <python_script> [optional_args]"
    exit 1
fi

PYTHON_SCRIPT="$1"
shift  # Remove first argument, keep the rest for passing to Python script

# Extract script name for job naming
SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)

#SBATCH --job-name=${SCRIPT_NAME}
#SBATCH --output=exec_logs/slurm_${SCRIPT_NAME}_%j.out
#SBATCH --error=exec_logs/slurm_${SCRIPT_NAME}_%j.err

# Create output directory if it doesn't exist
mkdir -p out
mkdir -p exec_logs

echo "========================================"
echo "Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Python script: $PYTHON_SCRIPT"
echo "Additional args: $@"
echo "========================================"
echo ""

# Load required modules (adjust for your cluster)
echo "Loading modules..."
module purge
module load 2025
module load Anaconda3/2025.06-1

# Activate conda environment
echo "Activating greendigit..."
source activate greendigit

# Verify environment
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
python --version
echo ""

# Check GPU availability
echo "GPU Information:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap,driver_version --format=csv
echo ""

# SLURM copies the script to a staging dir before running it, so BASH_SOURCE[0]
# resolves there instead of the project. SLURM_SUBMIT_DIR is the directory where
# sbatch was called from, which must be the project root.
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
cd "$SCRIPT_DIR" || exit 1
echo "Working directory: $(pwd)"

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# TabPFN authentication (required for non-interactive environments)
# Get your token from https://ux.priorlabs.ai/account
export TABPFN_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiNDVkNTk3YWYtZDg2OS00NjMyLThiYjgtZGZiMzVhMTc3MzAyIiwiZXhwIjoxODEwNDUyMDc2fQ.fqIIV5l2lgPKNPr4uH_zGdNwAORnb691Wi9DWdh0Zr8"

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

echo "========================================"
echo "Starting Experiment"
echo "========================================"
echo ""

# Build absolute path without relying on realpath (which fails if file not yet visible)
PYTHON_SCRIPT_ABS="$SCRIPT_DIR/$PYTHON_SCRIPT"
if [ ! -f "$PYTHON_SCRIPT_ABS" ]; then
    echo "Error: script not found at $PYTHON_SCRIPT_ABS"
    echo "SCRIPT_DIR=$SCRIPT_DIR"
    echo "PYTHON_SCRIPT=$PYTHON_SCRIPT"
    exit 1
fi
srun python "$PYTHON_SCRIPT_ABS" "$@"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job Completed"
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "========================================"

exit $EXIT_CODE