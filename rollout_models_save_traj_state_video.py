import os
import re
import argparse
import numpy as np

import gymnasium as gym
import metaworld

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Video deps are optional
try:
    import imageio.v2 as imageio  # noqa: F401
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}


# -------------------------
# Robust Gymnasium wrapper for Meta-World (handles 4 or 5 return step)
# -------------------------
class MetaWorldRobustGymnasiumEnv(gym.Env):
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

        out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def step(self, action):
        self._elapsed_steps += 1
        out = self._env.step(action)

        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        elif isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = bool(terminated)
            truncated = bool(truncated)
        else:
            raise RuntimeError(f"Unexpected step() return: {out}")

        # Enforce time-limit if needed
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


# -------------------------
# Filename parsing
# -------------------------
def parse_model_name(path: str):
    """
    Supports:
      <task>__<algo>__seed<id>.zip  e.g. push-v3__sac__seed0.zip
      <task>_<algo>_seed<id>.zip    e.g. push_sac_seed0.zip
    Returns (task, algo, seed)
    """
    base = os.path.basename(path)
    base = base[:-4] if base.endswith(".zip") else base

    m = re.match(r"^(?P<task>.+)__(?P<algo>sac|td3|ppo)__seed(?P<seed>\d+)$", base, re.IGNORECASE)
    if m:
        return m.group("task"), m.group("algo").lower(), int(m.group("seed"))

    m = re.match(r"^(?P<task>.+)_(?P<algo>sac|td3|ppo)_seed(?P<seed>\d+)$", base, re.IGNORECASE)
    if m:
        return m.group("task"), m.group("algo").lower(), int(m.group("seed"))

    raise ValueError(f"Bad model filename (expected __ or _ patterns): {path}")


def map_task_name(task: str):
    # If your filenames sometimes use short task names
    task_map = {"reach": "reach-v3", "push": "push-v3", "pick-place": "pick-place-v3"}
    return task_map.get(task, task)


# -------------------------
# Model loading
# -------------------------
def load_model(algo: str, zip_path: str, venv: DummyVecEnv, force_cpu_for_ppo: bool):
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo: {algo}")
    if algo == "ppo" and force_cpu_for_ppo:
        return ALGOS[algo].load(zip_path, env=venv, device="cpu")
    return ALGOS[algo].load(zip_path, env=venv, device="auto")


# -------------------------
# State capture (native mujoco backend)
# -------------------------
def get_qpos_qvel_native(mw_env):
    """
    Your env exposes:
      env.data: mujoco._structs.MjData
      env.model: mujoco._structs.MjModel
    So we record qpos/qvel from env.data.
    """
    if hasattr(mw_env, "data") and hasattr(mw_env.data, "qpos") and hasattr(mw_env.data, "qvel"):
        return mw_env.data.qpos.copy(), mw_env.data.qvel.copy(), "mw_env.data"
    return None, None, "no_mw_env.data.(qpos,qvel)"


# -------------------------
# Video frame capture (best-effort)
# -------------------------
def _to_uint8(frame):
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.size == 0:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    return None


def try_render_frame(wrapper_env: MetaWorldRobustGymnasiumEnv, mw_env):
    # 1) wrapper render
    try:
        f = _to_uint8(wrapper_env.render())
        if f is not None:
            return f, "wrapper.render()"
    except Exception:
        pass

    # 2) underlying env render(mode="rgb_array")
    try:
        f = _to_uint8(mw_env.render(mode="rgb_array"))
        if f is not None:
            return f, "mw_env.render(mode='rgb_array')"
    except Exception:
        pass

    return None, "no_frame"


# -------------------------
# Save helpers
# -------------------------
def save_episode_npz(path, meta: dict, obs, acts, rews, dones):
    np.savez_compressed(
        path,
        task=np.array(meta["task"]),
        algo=np.array(meta["algo"]),
        seed=np.int32(meta["seed"]),
        episode_index=np.int32(meta["episode"]),
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(acts, dtype=np.float32),
        rewards=np.asarray(rews, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
    )


def save_episode_states_npz(path, meta: dict, qpos_list, qvel_list, source: str):
    np.savez_compressed(
        path,
        task=np.array(meta["task"]),
        algo=np.array(meta["algo"]),
        seed=np.int32(meta["seed"]),
        episode_index=np.int32(meta["episode"]),
        source=np.array(source),
        qpos=np.asarray(qpos_list, dtype=np.float64),
        qvel=np.asarray(qvel_list, dtype=np.float64),
    )


# -------------------------
# Rollout one model
# -------------------------
def rollout_one_model(
    zip_path: str,
    task_id: str,
    algo: str,
    seed: int,
    episodes: int,
    max_steps: int,
    out_traj_dir: str,
    out_state_dir: str | None,
    out_vid_dir: str | None,
    fps: int,
    rollout_seed: int,
    force_cpu_for_ppo: bool,
    deterministic: bool,
):
    os.makedirs(out_traj_dir, exist_ok=True)
    if out_state_dir is not None:
        os.makedirs(out_state_dir, exist_ok=True)
    if out_vid_dir is not None:
        os.makedirs(out_vid_dir, exist_ok=True)

    mt1 = metaworld.MT1(task_id)
    env_cls = mt1.train_classes[task_id]
    mw_env = env_cls()
    mw_env.set_task(mt1.train_tasks[0])

    wrapper = MetaWorldRobustGymnasiumEnv(mw_env, render_mode="rgb_array")
    venv = DummyVecEnv([lambda: wrapper])

    model = load_model(algo, zip_path, venv, force_cpu_for_ppo=force_cpu_for_ppo)

    rng = np.random.default_rng(rollout_seed)
    run_name = f"{task_id}__{algo}__seed{seed}"

    can_video = (out_vid_dir is not None) and HAS_IMAGEIO
    if (out_vid_dir is not None) and (not HAS_IMAGEIO):
        print(f"[VIDEO][{run_name}] imageio not installed -> skipping videos.")

    for ep in range(episodes):
        # Randomize task instance for diversity
        mw_env.set_task(mt1.train_tasks[int(rng.integers(0, len(mt1.train_tasks)))])

        obs = venv.reset()  # (1, obs_dim)

        obs_list, act_list, rew_list, done_list = [], [], [], []

        # State capture (native mujoco)
        qpos_list, qvel_list = [], []
        state_source = None
        if out_state_dir is not None:
            qpos, qvel, src = get_qpos_qvel_native(mw_env)
            if qpos is None:
                print(f"[STATE][{run_name} ep{ep:03d}] Cannot record state: {src}")
            else:
                state_source = src
                qpos_list.append(qpos)
                qvel_list.append(qvel)

        # Video capture
        frames = [] if can_video else None
        frame_sources = {}
        if frames is not None:
            f, src = try_render_frame(wrapper, mw_env)
            if f is not None:
                frames.append(f)
                frame_sources[src] = frame_sources.get(src, 0) + 1

        for _t in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, rewards, dones, infos = venv.step(action)

            obs_list.append(obs[0].copy())
            act_list.append(action[0].copy())
            rew_list.append(float(rewards[0]))
            done_list.append(bool(dones[0]))

            obs = next_obs

            if state_source is not None:
                qpos, qvel, _ = get_qpos_qvel_native(mw_env)
                if qpos is not None:
                    qpos_list.append(qpos)
                    qvel_list.append(qvel)

            if frames is not None:
                f, src = try_render_frame(wrapper, mw_env)
                if f is not None:
                    frames.append(f)
                    frame_sources[src] = frame_sources.get(src, 0) + 1

            if dones[0]:
                break

        # Save trajectory
        ep_npz = os.path.join(out_traj_dir, f"{run_name}__ep{ep:03d}.npz")
        save_episode_npz(
            ep_npz,
            {"task": task_id, "algo": algo, "seed": seed, "episode": ep},
            obs_list, act_list, rew_list, done_list,
        )

        # Save state (if captured)
        if out_state_dir is not None and state_source is not None:
            ep_state = os.path.join(out_state_dir, f"{run_name}__ep{ep:03d}__state.npz")
            save_episode_states_npz(
                ep_state,
                {"task": task_id, "algo": algo, "seed": seed, "episode": ep},
                qpos_list,
                qvel_list,
                state_source,
            )

        # Save video (best-effort)
        if frames is not None:
            if len(frames) == 0:
                print(f"[VIDEO][{run_name} ep{ep:03d}] 0 frames -> no mp4 (render not working here).")
            else:
                mp4_path = os.path.join(out_vid_dir, f"{run_name}__ep{ep:03d}.mp4")
                try:
                    import imageio.v2 as imageio
                    imageio.mimsave(mp4_path, frames, fps=fps)
                    src_summary = ", ".join(f"{k}:{v}" for k, v in sorted(frame_sources.items()))
                    print(f"[VIDEO][{run_name} ep{ep:03d}] saved {len(frames)} frames -> {mp4_path} (sources: {src_summary})")
                except Exception as e:
                    print(f"[VIDEO][{run_name} ep{ep:03d}] FAILED mp4 write: {e}")

    venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--outdir", default="rollouts_out")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=150)

    ap.add_argument("--save_states", action="store_true", help="Save qpos/qvel per step (works headless)")
    ap.add_argument("--save_videos", action="store_true", help="Best-effort MP4 (likely fails on your cluster)")
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--rollout_seed", type=int, default=12345)
    ap.add_argument("--force_cpu_for_ppo", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions for predict()")

    args = ap.parse_args()

    models = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir) if f.endswith(".zip")]
    models.sort()

    traj_dir = os.path.join(args.outdir, "trajectories")
    state_dir = os.path.join(args.outdir, "states") if args.save_states else None
    vid_dir = os.path.join(args.outdir, "videos") if args.save_videos else None

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    if state_dir is not None:
        os.makedirs(state_dir, exist_ok=True)
    if vid_dir is not None:
        os.makedirs(vid_dir, exist_ok=True)

    print(f"Found {len(models)} model zips in {args.models_dir}")

    for zp in models:
        task, algo, seed = parse_model_name(zp)
        task_id = map_task_name(task)

        print(f"\n=== Rollout {os.path.basename(zp)} -> task={task_id}, algo={algo}, seed={seed} ===")
        rollout_one_model(
            zip_path=zp,
            task_id=task_id,
            algo=algo,
            seed=seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            out_traj_dir=traj_dir,
            out_state_dir=state_dir,
            out_vid_dir=vid_dir,
            fps=args.fps,
            rollout_seed=args.rollout_seed + seed,
            force_cpu_for_ppo=args.force_cpu_for_ppo,
            deterministic=args.deterministic,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()