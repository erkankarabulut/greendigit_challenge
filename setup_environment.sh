#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=setup_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=setup_output_%j.log
#SBATCH --error=setup_error_%j.err

echo "=== Environment Setup Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load required modules (adjust for your cluster)
echo "Loading modules..."
module purge
module load 2025
module load Anaconda3/2025.06-1

# Display loaded modules
echo "Loaded modules:"
module list
echo ""

# Accept conda Terms of Service
echo "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
echo "Creating conda environment 'greendigit'..."
conda create -n greendigit python=3.11 -y

# Activate environment
echo "Activating environment..."
source activate greendigit

# Verify Python version
echo "Python version:"
python --version
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with the correct CUDA build for the cluster
# Adjust the --index-url if your CUDA version differs
echo "Installing PyTorch (CUDA 12.1 build)..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install all remaining dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Login to Hugging Face (required for TabPFN)
echo ""
echo "NOTE: TabPFN requires Hugging Face authentication."
echo "Please run: huggingface-cli login"
echo "Or set the HF_TOKEN environment variable before running experiments."

# Verify key packages
echo ""
echo "=== Verifying Installation ==="
python -c "
import sys
print(f'Python: {sys.version}')

packages = [
    ('torch',           lambda m: f'{m.__version__} (CUDA={m.cuda.is_available()})'),
    ('numpy',           lambda m: m.__version__),
    ('pandas',          lambda m: m.__version__),
    ('sklearn',         lambda m: m.__version__),
    ('xgboost',         lambda m: m.__version__),
    ('tabpfn',          lambda m: m.__version__),
    ('tabpfn_time_series', lambda m: m.__version__),
    ('task_a',          lambda m: 'ok'),
    ('dirac_sim',       lambda m: 'ok'),
]

for name, version_fn in packages:
    try:
        import importlib
        mod = importlib.import_module(name)
        print(f'  {name:<25} {version_fn(mod)}')
    except ImportError as e:
        print(f'  {name:<25} MISSING: {e}')
"

echo ""
echo "=== Environment Setup Completed ==="
echo "Date: $(date)"
echo ""
echo "To use this environment in your jobs, add to your job script:"
echo "  module load 2025"
echo "  module load Anaconda3/2025.06-1"
echo "  source activate greendigit"