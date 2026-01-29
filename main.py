import yaml
import argparse
from runner import (
    llm_num_optim_runner,
    llm_num_optim_semantics_runner,
    llm_num_reward_runner,
    llm_traj_comparator_runner
)
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Optional override for logdir from the config file",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional override for summary flag from the config file",
    )
    parser.add_argument(
        'overrides', 
        nargs=argparse.REMAINDER, 
        help="Other config overrides (e.g., key=value)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Allow command-line override of logdir used by SLURM job script
    if args.logdir is not None:
        config["logdir"] = args.logdir
    else:
        os.makedirs(config["logdir"], exist_ok=True)
        
    if args.summary is not None:
        # Convert the string "true" or "false" to a Python boolean
        config["summary"] = args.summary.lower() in ['true', '1', 't']
        print(f"--- Overriding config: summary = {config['summary']} ---")
        
    for override in args.overrides:
        if '=' not in override:
            print(f"Warning: Skipping invalid override: {override}", file=sys.stderr)
            continue
        
        # Split on the first '=' only
        key, value = override.split('=', 1)
        
        if key in config:
            # Try to cast the value to the original type (bool, int, float)
            try:
                original_type = type(config[key])
                if original_type == bool:
                    value = value.lower() in ['true', '1', 't']
                elif original_type is not None and original_type in [int, float]:
                     value = original_type(value)
                # Otherwise, keep as string
            except (ValueError, TypeError):
                pass # Keep as string if cast fails
            
            print(f"--- Overriding config: {key} = {value} ---")
            config[key] = value
        else:
            print(f"Warning: '{key}' not found in base config. Skipping.", file=sys.stderr)
            
    # print("Final configuration:")
    # for k, v in config.items():
    #     print(f"  {k}: {v}")

    llm_traj_comparator_runner.run_training_loop(**config)


    # if config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
    #     llm_num_optim_runner.run_training_loop(**config)
    # elif config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
    #     llm_num_optim_semantics_runner.run_training_loop(**config)
    # elif config["task"] in ["dist_state_llm_num_reward", "cont_state_llm_num_reward"]:
    #     llm_num_reward_runner.run_training_loop(**config)
    # else:
    #     raise ValueError(f"Task {config['task']} not recognized.")


if __name__ == "__main__":
    main()
