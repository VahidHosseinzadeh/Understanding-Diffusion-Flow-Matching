#!/bin/bash
# Flow matching training, as a Slurm job.
#
#   sbatch train.sbatch                      # submit with all defaults
#   squeue -u $USER                          # watch the queue
#   tail -f slurm_dfm_<jobid>.log            # watch the run
#   scancel <jobid>                          # kill it
#
# ---------------------------------------------------------------------
# KNOBS -- set any of these as environment variables in front of sbatch.
# Nothing in this file needs editing to change an experiment.
#
#   TARGET       velocity | x_data | noise           (default: velocity)
#   FM_PATH      linear                              (default: linear)
#   SAMPLER      euler | heun                        (default: euler)
#   DATA         fashion_mnist | moons |
#                eight_gaussians | spiral            (default: fashion_mnist)
#   MODEL        unet | mlp        (default: empty = auto, mlp for 2D data,
#                                   unet for images -- usually leave it alone)
#   T_DIST       uniform | logit_normal              (default: uniform)
#   EPOCHS                                           (default: 30)
#   BATCH_SIZE                                       (default: 128)
#   SEED                                             (default: 0)
#   DIAGNOSTICS  1 to log loss-vs-t, velocity-norm profile
#                and a solver x budget matrix     (default: off, ~5s/preview)
#   EXPERIMENT   wandb group tying a set of runs together (default: baseline)
#   EXTRA        any other train.py flag, passed through verbatim
#
# NOTE: the path knob is FM_PATH, not PATH -- PATH is the shell's
# executable search path, and overwriting it would break this script.
#
# EXAMPLES
#   TARGET=noise sbatch train.sbatch
#   TARGET=x_data EPOCHS=50 sbatch train.sbatch
#   DATA=moons sbatch train.sbatch                   # model auto-switches to mlp
#   EXTRA="--lr 1e-4 --base-channels 32" sbatch train.sbatch
#   DIAGNOSTICS=1 TARGET=noise sbatch train.sbatch   # see the boundary blow-up
#
#   # the comparison the three-axis split exists for -- one run per target,
#   # all in one wandb group so their curves overlay:
#   for tg in velocity x_data noise; do
#       TARGET=$tg EXPERIMENT=target-sweep sbatch train.sbatch
#   done
#
# TO RUN BY HAND instead (in an `srun` session, never on the login node):
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G \
#        --time=2:00:00 --pty bash
#   python dfm/train.py --data fashion_mnist --target noise --epochs 30 \
#       --batch-size 128 --num-workers 8 --tracker wandb \
#       --wandb-group manual --out-dir runs/manual
# ---------------------------------------------------------------------
#
#SBATCH --job-name=dfm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # this cluster wants --gres, not --gpus
#SBATCH --time=04:00:00              # partition caps at 24h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_%x_%j.log     # %x = job name, %j = job id

set -euo pipefail

# --- knobs, with defaults --------------------------------------------
TARGET="${TARGET:-velocity}"
FM_PATH="${FM_PATH:-linear}"
SAMPLER="${SAMPLER:-euler}"
DATA="${DATA:-fashion_mnist}"
MODEL="${MODEL:-}"                   # empty = let train.py choose
T_DIST="${T_DIST:-uniform}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-0}"
EXPERIMENT="${EXPERIMENT:-baseline}"
DIAGNOSTICS="${DIAGNOSTICS:-}"
EXTRA="${EXTRA:-}"

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
# Run name and out-dir both carry the varying knobs, so two runs of the
# same sweep never collide on disk or become indistinguishable in wandb.
RUN_ID="${DATA}_${FM_PATH}_${TARGET}_s${SEED}_${SLURM_JOB_ID}"

ARGS=(
    --data "$DATA"
    --path "$FM_PATH"
    --target "$TARGET"
    --sampler "$SAMPLER"
    --t-dist "$T_DIST"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --seed "$SEED"
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
    --tracker wandb
    --wandb-group "$EXPERIMENT"
    --wandb-name "$RUN_ID"
    --out-dir "runs/$RUN_ID"
)
if [ -n "$MODEL" ]; then ARGS+=(--model "$MODEL"); fi
if [ -n "$DIAGNOSTICS" ]; then ARGS+=(--diagnostics); fi

echo "+ python dfm/train.py ${ARGS[*]} $EXTRA"
# $EXTRA is deliberately unquoted so it splits into separate flags.
python dfm/train.py "${ARGS[@]}" $EXTRA

echo "done."
