#!/bin/bash
# ----------------------------------------------------------------              
# SBATCH HEADERS: These tell the scheduler what hardware we need.
# ----------------------------------------------------------------              
#SBATCH -J stage2_json           # Sets the name of the job in the queue
#SBATCH -o stage2_out_%j.txt     # Where all printed output and errors go
#SBATCH -p gpu1v100              # The specific partition (hardware group)
#SBATCH --gres=gpu:1             # CRITICAL: Actually attaches the GPU to my job
#SBATCH -N 1                     # Requests 1 physical compute node
#SBATCH -n 1                     # Requests 1 CPU core (enough for standard Python)
#SBATCH -t 12:00:00              # Time limit (HH:MM:SS) - job kills after this

# ----------------------------------------------------------------
# SOFTWARE SETUP: Hooking up the drivers and libraries.
# ----------------------------------------------------------------
# Loads the system CUDA drivers so my GPU can talk to Python
module load cudatoolkit/12.5.0_555.42.02 

# Optional: Helps TensorFlow/PyTorch find specific GPU libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Set pythonpath so config can be imported securely when running from root
export PYTHONPATH=.

# Activate my python environment
# Note: Adjust if I end up using conda (e.g. source activate myenv)
source .venv/bin/activate

# Execute Stage 2 (JSON Specialist Training)
python training/train_stage2.py
