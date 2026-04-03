#!/bin/bash

module load ollama/0.12.3
export OLLAMA_MODULES=/data/datasets/community/ollama
OLLAMA_CONTEXT_LENGTH=131072 ollama serve


module load mamba/latest
source activate thesis
module load ollama/0.12.3
export OLLAMA_HOST=10.139.126.15:11434

ollama-start

python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml
python3 main.py --config configs/mountaincar/mountaincar_propsp.yaml

python3 main.py --config configs/invertedpendulum/invertedpendulum_propsr.yaml
python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_propsr.yaml
python3 main.py --config configs/mountaincar/mountaincar_propsr.yaml

python3 main.py --config configs/hopper/hopper_trajectory.yaml
python3 main.py --config configs/invertedpendulum/invertedpendulum_trajectory.yaml
python3 main.py --config configs/mountaincar/mountaincar_trajectory.yaml

python3 main.py --config configs/meta-reacher/reacher_trajectory.yaml

python main/train_verdict.py --env reach-v3 --use_llm_pref --wandb --run_name verdict-reach-v3-seed0 --device cpu --config configs/meta-reacher/reacher_trajectory.yaml

python hill-climbing/train.py --env Hopper-v5 --use_llm_pref --wandb --run_name hopper-hill-climb-seed-0 --device cpu --config configs/hopper/hopper_trajectory.yaml

watch -n 1 -t "myjobs | grep -Ec '^[[:space:]]*[0-9]'"