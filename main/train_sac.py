import argparse
import os

import gymnasium as gym
import metaworld  # noqa: F401
import wandb

from stable_baselines3 import SAC

from core.utils import set_seed
from core.callbacks import WandbEvalCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="reach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="sac-reach-v3-seed0")
    parser.add_argument("--project", type=str, default="llm-sac-metaworld")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args),
            group="vanilla"
        )

    if args.env in metaworld.ML1.ENV_NAMES:
        env = gym.make("Meta-World/MT1", env_name=args.env)
        eval_env = gym.make("Meta-World/MT1", env_name=args.env)
    else:
        env = gym.make(args.env)
        eval_env = gym.make(args.env)

    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    eval_callback = WandbEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_wandb=args.wandb,
        prefix="vanilla",
        is_mtl=args.env in metaworld.ML1.ENV_NAMES,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
    )

    save_path = os.path.join(args.save_dir, f"sac_{args.env}_seed{args.seed}")
    model.save(save_path)
    print(f"Saved model to {save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()