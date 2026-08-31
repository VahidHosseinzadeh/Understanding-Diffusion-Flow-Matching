#!/bin/bash
#SBATCH --job-name=dfm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # this cluster wants --gres, not --gpus
#SBATCH --time=04:00:00              # partition caps at 24h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.log     # %x = job name, %j = job id

set -euo pipefail

# --- environment -----------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

# No activation line: Slurm exports your submitting environment by default
# (--export=ALL), so the conda env that is active when you run `sbatch` is
# still active here. Submit from a shell showing (torch2.2) in the prompt.
# If that ever stops holding, uncomment:
# source activate torch2.2
#
# No install needed: every module sits in dfm/, beside train.py.

python -c "import torch; print('torch', torch.__version__)" || {
    echo "wrong environment: torch not importable" >&2; exit 1
}

# --- data + tracking -------------------------------------------------
# Fashion-MNIST must already be in data/. Pre-download once on the login node
# (torchvision downloads on first use, which fails if the node is offline):
#   PYTHONPATH=dfm python -c "from dataset import get_fashion_mnist; get_fashion_mnist()"
#
# wandb auth comes from ~/.netrc, written once by `wandb login` -- it persists
# across sessions and is visible on compute nodes because $HOME is shared.
# If a job ever hangs at "wandb: Currently logged in as...", the node has no
# outbound network; uncomment the next line and `wandb sync` from a login node.
# export WANDB_MODE=offline
export TQDM_MININTERVAL=10       # keep the log file from filling with progress bars

echo "node=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PYTHONPATH=dfm python -c "from utils import get_device; print('device:', get_device())"

# --- the run ---------------------------------------------------------
python dfm/train.py \
    --data fashion_mnist \
    --model unet \
    --epochs 30 \
    --batch-size 128 \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    --tracker wandb \
    --wandb-group "${SLURM_JOB_NAME}-${SLURM_JOB_ID}" \
    --out-dir "runs/fashion_mnist_${SLURM_JOB_ID}"

echo "done."
