#!/bin/bash

module load ollama/0.12.3
export OLLAMA_MODULES=/data/datasets/community/ollama
OLLAMA_CONTEXT_LENGTH=131072 ollama serve

export OLLAMA_HOST=10.139.126.15:11434

module load mamba/latest
source activate thesis
ollama-start

python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml
python3 main.py --config configs/mountaincar/mountaincar_propsp.yaml

python3 main.py --config configs/invertedpendulum/invertedpendulum_propsr.yaml
python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_propsr.yaml
python3 main.py --config configs/mountaincar/mountaincar_propsr.yaml

python3 main.py --config configs/hopper/hopper_trajectory.yaml

watch -n 1 -t "myjobs | grep -Ec '^[[:space:]]*[0-9]'"