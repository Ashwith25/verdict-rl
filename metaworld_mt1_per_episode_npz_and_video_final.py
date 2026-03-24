"""
metaworld_mt1_per_episode_npz_and_video_final.py

This version is robust to Meta-World returning either:
- old-gym step:      obs, reward, done, info
- gymnasium step:    obs, reward, terminated, truncated, info

and always exposes Gymnasium API outward:
- reset() -> (obs, info)
- step()  -> (obs, reward, terminated, truncated, info)

Works with SB3 that is Gymnasium-based (your stack: Monitor expects gymnasium).

Outputs:
- models/*.zip
- trajectories/<task>__<algo>__seed<seed>__epXYZ.npz
- videos/<task>__<algo>__seed<seed>__epXYZ.mp4 (optional)
"""

import os
import json
import argparse
import numpy as np

import gymnasium as gym
import metaworld

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


class MetaWorldRobustGymnasiumEnv(gym.Env):
    """
    Robust Gymnasium wrapper for Meta-World.

    Exposes:
      reset() -> (obs, info)
      step()  -> (obs, reward, terminated, truncated, info)

    Accepts underlying Meta-World that may return either 4-tuple or 5-tuple from step().
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, mw_env, render_mode="rgb_array"):
        super().__init__()
        self._env = mw_env
        self.render_mode = render_mode

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._max_episode_steps = int(getattr(self._env, "max_path_length", 150))
        self._elapsed_steps = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._elapsed_steps = 0

        # Meta-World reset is typically old-gym: obs = env.reset()
        out = self._env.reset()
        # Some forks may return (obs, info) already; support it:
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def step(self, action):
        self._elapsed_steps += 1

        out = self._env.step(action)

        # Underlying may be 4-tuple (old gym) or 5-tuple (gymnasium)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        elif isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = bool(terminated)
            truncated = bool(truncated)
        else:
            raise RuntimeError(f"Unexpected step() return from Meta-World: type={type(out)} value={out}")

        # Enforce time limit if underlying doesn't handle it cleanly
        if (not terminated) and (not truncated) and (self._elapsed_steps >= self._max_episode_steps):
            truncated = True

        return obs, float(reward), terminated, truncated, info

    def render(self):
        # Try common render signatures
        try:
            return self._env.render(mode=self.render_mode)
        except TypeError:
            try:
                return self._env.render()
            except Exception:
                return None

    def close(self):
        self._env.close()


def algo_factory(algo_name: str):
    a = algo_name.lower()
    if a == "sac":
        return SAC
    if a == "td3":
        return TD3
    if a == "ppo":
        return PPO
    raise ValueError(f"Unknown algo: {algo_name}")


def unwrap_to_base_env(env):
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def get_mw_env_from_venv(venv: DummyVecEnv):
    env0 = venv.envs[0]
    base = unwrap_to_base_env(env0)  # MetaWorldRobustGymnasiumEnv
    return base._env


def make_mt1_venv(task_name: str, seed: int):
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]

    mw_env = env_cls()
    mw_env.set_task(mt1.train_tasks[0])

    env = MetaWorldRobustGymnasiumEnv(mw_env, render_mode="rgb_array")
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])

    set_random_seed(seed)
    return venv, mt1


def train_policy(algo_name: str, venv, seed: int, total_steps: int, log_dir: str):
    Algo = algo_factory(algo_name)
    a = algo_name.lower()

    if a in ["sac", "td3"]:
        model = Algo(
            "MlpPolicy",
            venv,
            verbose=1,
            seed=seed,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=log_dir,
        )
    else:
        model = Algo(
            "MlpPolicy",
            venv,
            verbose=1,
            seed=seed,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            learning_rate=3e-4,
            tensorboard_log=log_dir,
        )

    model.learn(total_timesteps=total_steps, progress_bar=True)
    return model


def save_episode_npz(
    out_path: str,
    *,
    task_name: str,
    algo: str,
    seed: int,
    episode_index: int,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
):
    np.savez_compressed(
        out_path,
        task_name=np.array(task_name),
        algo=np.array(algo),
        seed=np.int32(seed),
        episode_index=np.int32(episode_index),
        observations=observations.astype(np.float32, copy=False),
        actions=actions.astype(np.float32, copy=False),
        rewards=rewards.astype(np.float32, copy=False),
        dones=dones.astype(np.bool_, copy=False),
    )


def try_render_rgb(mw_env):
    try:
        return mw_env.render(mode="rgb_array")
    except TypeError:
        try:
            return mw_env.render()
        except Exception:
            return None
    except Exception:
        return None


def rollout_and_save_per_episode(
    *,
    model,
    venv: DummyVecEnv,
    mt1,
    task_name: str,
    algo_name: str,
    policy_seed: int,
    rollout_seed: int,
    n_episodes: int,
    max_steps: int,
    traj_out_dir: str,
    video_out_dir: str | None,
    fps: int,
):
    os.makedirs(traj_out_dir, exist_ok=True)

    imageio = None
    if video_out_dir is not None:
        os.makedirs(video_out_dir, exist_ok=True)
        import imageio.v2 as imageio  # noqa: F401
        import imageio.v2 as imageio

    rng = np.random.default_rng(rollout_seed)
    mw_env = get_mw_env_from_venv(venv)
    run_name = f"{task_name}__{algo_name}__seed{policy_seed}"

    for ep in range(n_episodes):
        task = mt1.train_tasks[int(rng.integers(0, len(mt1.train_tasks)))]
        mw_env.set_task(task)

        obs = venv.reset()  # batched obs: (1, obs_dim)

        obs_list, act_list, rew_list, done_list = [], [], [], []
        frames = [] if video_out_dir is not None else None

        if frames is not None:
            f0 = try_render_rgb(mw_env)
            if f0 is not None:
                frames.append(f0)

        for _t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, rewards, dones, infos = venv.step(action)

            obs_list.append(obs[0].copy())
            act_list.append(action[0].copy())
            rew_list.append(float(rewards[0]))
            done_list.append(bool(dones[0]))

            obs = next_obs

            if frames is not None:
                f = try_render_rgb(mw_env)
                if f is not None:
                    frames.append(f)

            if dones[0]:
                break

        ep_npz = os.path.join(traj_out_dir, f"{run_name}__ep{ep:03d}.npz")
        save_episode_npz(
            ep_npz,
            task_name=task_name,
            algo=algo_name,
            seed=policy_seed,
            episode_index=ep,
            observations=np.asarray(obs_list),
            actions=np.asarray(act_list),
            rewards=np.asarray(rew_list),
            dones=np.asarray(done_list),
        )

        if frames is not None and len(frames) > 0:
            ep_mp4 = os.path.join(video_out_dir, f"{run_name}__ep{ep:03d}.mp4")
            try:
                imageio.mimsave(ep_mp4, frames, fps=fps)
            except Exception as e:
                print(f"[WARN] Failed to save video {ep_mp4}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="runs_metaworld_mt1")
    parser.add_argument("--tasks", type=str, nargs="+", default=["reach-v3", "push-v3", "pick-place-v3"])
    parser.add_argument("--total_steps", type=int, default=600_000)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--video_fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models_dir = os.path.join(args.outdir, "models")
    traj_dir = os.path.join(args.outdir, "trajectories")
    vid_dir = os.path.join(args.outdir, "videos")
    tb_dir = os.path.join(args.outdir, "tb")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    if args.save_videos:
        os.makedirs(vid_dir, exist_ok=True)

    policy_specs = [
        ("sac", 0), ("sac", 1), ("sac", 2), ("sac", 3),
        ("td3", 0), ("td3", 1),
        ("ppo", 0), ("ppo", 1),
    ]

    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(
            {
                "tasks": args.tasks,
                "total_steps": args.total_steps,
                "episodes_per_policy": args.episodes,
                "max_steps_per_episode": args.max_steps,
                "policy_specs": policy_specs,
                "videos_enabled": bool(args.save_videos),
                "video_fps": args.video_fps,
            },
            f,
            indent=2,
        )

    for task_name in args.tasks:
        print(f"\n==== Task: {task_name} ====")
        for algo_name, seed in policy_specs:
            run_name = f"{task_name}__{algo_name}__seed{seed}"
            print(f"\n--- Training {run_name} ---")

            venv, mt1 = make_mt1_venv(task_name, seed=seed)
            steps = args.total_steps if algo_name.lower() != "ppo" else int(args.total_steps * 1.5)

            model = train_policy(algo_name=algo_name, venv=venv, seed=seed, total_steps=steps, log_dir=tb_dir)

            model_path = os.path.join(models_dir, run_name)
            model.save(model_path)
            print(f"Saved model: {model_path}.zip")

            print(f"--- Rollout + save per-episode NPZ{' + MP4' if args.save_videos else ''} for {run_name} ---")
            rollout_and_save_per_episode(
                model=model,
                venv=venv,
                mt1=mt1,
                task_name=task_name,
                algo_name=algo_name,
                policy_seed=seed,
                rollout_seed=seed + 10_000,
                n_episodes=args.episodes,
                max_steps=args.max_steps,
                traj_out_dir=traj_dir,
                video_out_dir=(vid_dir if args.save_videos else None),
                fps=args.video_fps,
            )

            venv.close()

    print("\nDone.")


if __name__ == "__main__":
    main()