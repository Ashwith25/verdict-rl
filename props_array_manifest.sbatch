#!/bin/bash
#SBATCH --job-name=propsR
#SBATCH --array=1-10                # Total number of sub-jobs (will be overridden by command line)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # Default cores per sub-job (can be overridden with -c)
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=general
#SBATCH --time=24:00:00             # Wall time for each sub-job
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --output=props-BT/slurm/task_%a.log
#SBATCH --error=props-BT/slurm/task_%a.err

# --- 1. Setup: Read Manifest ---
manifest="${1:?ERROR -- must pass a manifest file as first argument}"
taskid=$SLURM_ARRAY_TASK_ID

if [ ! -f "$manifest" ]; then
    echo "ERROR: Manifest file not found: $manifest"
    exit 1
fi

# Read the task ID line from manifest
# Format: <random_seed> <scaler> <model>
line=$(sed -n "${taskid}p" "$manifest")

if [ -z "$line" ]; then
    echo "ERROR: Task ID $taskid not found in manifest (manifest has $(wc -l < "$manifest") lines)"
    exit 1
fi

# Extract parameters from the line
random_seed=$(echo "$line" | cut -d ' ' -f 1)
config=$(echo "$line" | cut -d ' ' -f 2)
logdir=$(echo "$line" | cut -d ' ' -f 3)
summary=$(echo "$line" | cut -d ' ' -f 4)
max_traj_count=$(echo "$line" | cut -d ' ' -f 5)
max_best_length=$(echo "$line" | cut -d ' ' -f 6)

# --- Create output directory structure (like Python script does) ---
TRIAL_LOG_DIR="${logdir}/trial_${random_seed}"
mkdir -p "$TRIAL_LOG_DIR"

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

echo "========================================================================="
echo "SLURM Job Array Task: $taskid / $SLURM_ARRAY_JOB_ID"
echo "Parameters from manifest:"
echo "  Random Seed: $random_seed"
echo "  Config: $config"
echo "  Log Directory: $logdir"
echo "  Summary Flag: $summary"
echo "  Max Trajectory Count: $max_traj_count"
echo "  Max Best Length: $max_best_length"
echo "========================================================================="


/home/apoojar4/.conda/envs/thesis/bin/python3 main.py \
    --config "$config" \
    --logdir "$TRIAL_LOG_DIR" \
    max_best_length="$max_best_length"

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
