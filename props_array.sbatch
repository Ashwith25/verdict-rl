#!/bin/bash
#SBATCH --job-name=hill-climb-hopper
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2           # Default cores per sub-job (can be overridden with -c)
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time=24:00:00             # Wall time for each sub-job
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --output=hill-climb_logs/hopper/slurm/task_%a.log
#SBATCH --error=hill-climb_logs/hopper/slurm/task_%a.err

# --- Job Configuration ---
CONFIG_FILE="configs/hopper/hopper_trajectory.yaml"
taskid=$SLURM_ARRAY_TASK_ID
# BASE_LOG_DIR="verdict-rl/hopper-seed-0"

# --- Create a unique directory for this specific trial's output ---
# TRIAL_LOG_DIR="${BASE_LOG_DIR}/trial_${SLURM_ARRAY_TASK_ID}"
# mkdir -p "$BASE_LOG_DIR"

# --- Load Modules ---
module load mamba/latest
module load ollama/0.12.3

# --- START THE OLLAMA SERVER IN THE BACKGROUND ---
echo "--- Starting Ollama server on $(hostname) ---"

# 1. Get a random free port assigned by the OS
#    This ensures every job on the same node gets a unique port.
export OLLAMA_PORT=$(python3 -c 'import socket; 
for p in range(11434, 11535):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", p)) != 0:
            print(p)
            break
')

if [ -z "$OLLAMA_PORT" ]; then
    echo "ERROR: No free ports available in range 11434-11534 on this node."
    exit 1
fi

# 2. Configure Ollama to listen on this specific port
export OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"

echo "--- Assigned Port: $OLLAMA_PORT ---"
echo "--- OLLAMA_HOST set to: $OLLAMA_HOST ---"

OLLAMA_CONTEXT_LENGTH=131072 ollama serve &
OLLAMA_PID=$! 
sleep 15

echo "--- Checking Ollama status ---"
ollama list

# --- Activate Mamba/Conda Environment ---
source activate thesis

# /home/apoojar4/.conda/envs/thesis/bin/python3 main.py \
#     --config "$CONFIG_FILE" \
#     --logdir "$BASE_LOG_DIR"

/home/apoojar4/.conda/envs/thesis/bin/python3 hill-climbing/train.py \
    --env Hopper-v5 \
    --seed $taskid \
    --use_llm_pref \
    --wandb \
    --alpha 0.1 \
    --run_name hopper-hill-climb-seed-$taskid \
    --device cpu \
    --config "$CONFIG_FILE"

# Check exit status
EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "--- ERROR: Task $taskid failed with exit status $EXIT_STATUS ---"
else
    echo "--- Success: Task $taskid completed ---"
fi

# --- CLEAN UP: Stop the Ollama server ---
echo "--- Shutting down Ollama server (PID: $OLLAMA_PID) ---"
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

exit $EXIT_STATUS
