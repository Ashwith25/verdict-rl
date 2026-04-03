import os
import sys
import argparse
# Ensure project root is on sys.path so sibling packages (like `main`) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import metaworld
import yaml
import wandb
import weave

import numpy as np
import gymnasium as gym

from policy import get_action, rollout

from main.core.llm_comparator import LLMComparator
from main.core.utils import set_seed

def random_params(rank):
    return np.random.uniform(-1, 1, size=rank)

def mutate(params):
    return np.clip(params + np.random.normal(0, 0.3, size=params.shape[0]), -1, 1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="reach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--best_eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--best_n_eval_episodes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_llm_pref", action="store_true")
    parser.add_argument("--run_name", type=str, default="verdict-reach-v3-seed0")
    parser.add_argument("--project", type=str, default="llm-sac-metaworld")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["logdir"] = f"{config["logdir"]}_{args.seed}"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(config["logdir"], exist_ok=True)

    if args.wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args),
            group="verdict"
        )
        weave.init(project_name=args.project)
    
    if args.env in metaworld.ML1.ENV_NAMES:
        env = gym.make("Meta-World/MT1", env_name=args.env)
    else:
        env = gym.make(args.env)

    set_seed(args.seed)
    env.reset(seed=args.seed)

    best_params = random_params(rank=config["dim_states"] * config["dim_actions"] + config["dim_actions"])
    best_traj = rollout(env, config["dim_states"], config["dim_actions"], best_params)

    # print("Best Trajectory", best_traj)

    best_reward = sum([x["reward"] for x in best_traj])
    best_rewards = [best_reward]

    pref_model = LLMComparator(use_llm_pref=args.use_llm_pref, **config)

    with open(f"{config["logdir"]}/final_log.txt", "w") as f:
        f.write(f"Episode, New Reward, Best Reward, Update Status\n")

    for i in range(100):
        new_params = mutate(best_params)
        new_traj = rollout(env, config["dim_states"], config["dim_actions"], new_params)
        directory = f"episode_{i}"
        # prompt = build_prompt(
        #     summarize(new_traj),
        #     summarize(best_traj),
        # )

        winner = pref_model.compare(directory, new_traj, best_traj)
        print(winner)

        best_reward = sum([x["reward"] for x in best_traj])
        current_reward = sum([x["reward"] for x in new_traj])

        print(f"Iter {i+1} | Current Reward: {current_reward:.2f} | Best Reward: {best_reward:.2f} | Winner: {"Current" if winner == "new" else "Best"}")

        if winner == "new":
            best_params = new_params
            best_traj = new_traj
        
        best_rewards.append(best_reward)
        with open(f"{config["logdir"]}/final_log.txt", "a") as f:
            f.write(f"{i+1}, {current_reward:.2f}, {best_reward:.2f}, {'Updated' if winner == 'new' else 'Not Updated'}\n")

        log_data = {
                "hill/reward": best_reward,
                "timesteps": i,
            } 
        wandb.log(log_data, step=i)
    print("Done")
    print("Final Reward:", best_reward)


if __name__ == "__main__":
    main()