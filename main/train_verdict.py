import argparse
import os
import yaml

import gymnasium as gym
import metaworld
import wandb
import weave

from stable_baselines3 import SAC

from core.utils import set_seed
from core.reference_buffer import ReferenceBank
from core.llm_comparator import LLMComparator
from core.callbacks import WandbEvalCallback, PreferenceCallback, BestPolicyCallback


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
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

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
        eval_env = gym.make("Meta-World/MT1", env_name=args.env)
    else:
        env = gym.make(args.env)
        eval_env = gym.make(args.env)

    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 1)

    pref_model = LLMComparator(use_llm_pref=args.use_llm_pref, **config)
    ref_bank = ReferenceBank(max_size=100)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    best_model = SAC(
        "MlpPolicy",
        eval_env,
        verbose=0,
        seed=args.seed,
        device=args.device,
    )

    pref_callback = PreferenceCallback(
        pref_model=pref_model,
        ref_bank=ref_bank,
        alpha=args.alpha,
        use_wandb=args.wandb,
    )

    eval_callback = WandbEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_wandb=args.wandb,
        prefix="verdict",
        is_mtl=args.env in metaworld.ML1.ENV_NAMES,
    )

    best_callback = BestPolicyCallback(
        eval_env=eval_env,
        pref_model=pref_model,
        best_model=best_model,
        eval_freq=args.best_eval_freq,
        ref_bank=ref_bank,
        n_eval_episodes=args.best_n_eval_episodes,
        tolerance=0.02,
        use_wandb=args.wandb,
        is_mtl=args.env in metaworld.ML1.ENV_NAMES,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[pref_callback, eval_callback, best_callback],
    )

    save_path = os.path.join(args.save_dir, f"verdict_{args.env}_seed{args.seed}")
    model.save(save_path)
    print(f"Saved model to {save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()