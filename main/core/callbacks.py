import copy
import wandb
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from core.eval import evaluate, evaluate_for_non_mtl


class WandbEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq=5000,
        n_eval_episodes=10,
        use_wandb=False,
        prefix="eval",
        is_mtl=False,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        self.prefix = prefix
        self.is_mtl = is_mtl
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            metrics = evaluate(self.model, self.eval_env, self.n_eval_episodes) if self.is_mtl else evaluate_for_non_mtl(self.model, self.eval_env, self.n_eval_episodes)

            log_data = {
                f"{self.prefix}/reward_mean": metrics["reward_mean"],
                f"{self.prefix}/reward_std": metrics["reward_std"],
                f"{self.prefix}/success_rate": metrics["success_rate"],
                "timesteps": self.num_timesteps,
            } if self.is_mtl else {
                f"{self.prefix}/reward_mean": metrics["reward_mean"],
                f"{self.prefix}/reward_std": metrics["reward_std"],
                "timesteps": self.num_timesteps,
            }

            print(log_data)
            if self.use_wandb:
                wandb.log(log_data, step=self.num_timesteps)

        return True


class PreferenceCallback(BaseCallback):
    def __init__(self, pref_model, ref_bank, alpha=0.05, use_wandb=False):
        super().__init__()
        self.pref_model = pref_model
        self.ref_bank = ref_bank
        self.alpha = alpha
        self.use_wandb = use_wandb
        self.episode = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % 100 == 0:
            print(f"[PREF HEARTBEAT] step={self.num_timesteps}", flush=True)
        new_obs = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        actions = self.locals.get("actions", None)

        reward = float(rewards[0]) if np.ndim(rewards) > 0 else float(rewards)
        done = bool(dones[0]) if np.ndim(dones) > 0 else bool(dones)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        action = None
        if actions is not None:
            action = actions[0] if np.ndim(actions) > 1 else actions

        self.episode.append(
            {
                "obs": new_obs[0] if np.ndim(new_obs) > 1 else new_obs,
                "action": action,
                "reward": reward,
                "info": info,
            }
        )

        if done:
            print(f"[PREF DONE] step={self.num_timesteps}, episode_len={len(self.episode)}", flush=True)
            self._process_episode()
            self.episode = []

        return True

    def _process_episode(self):
        tau_new = copy.deepcopy(self.episode)
        tau_ref = self.ref_bank.sample()

        directory = f"episode/episode_{self.episode_count}"

        print(f"[PREF] step={self.num_timesteps} episode_len={len(tau_new)} entering _process_episode")

        if tau_ref is not None:
            print(f"[PREF] calling LLM compare at step={self.num_timesteps}")
            winner = self.pref_model.compare(directory, tau_new, tau_ref)
            print(f"[PREF] LLM returned winner={winner} at step={self.num_timesteps}")
            s_pref = +1 if winner == "new" else -1
        else:
            print(f"[PREF] no reference trajectory yet at step={self.num_timesteps}")
            s_pref = 0

        T = len(tau_new)
        bonus = self.alpha * s_pref / max(T, 1)

        buffer = self.model.replay_buffer

        for i in range(T):
            idx = (buffer.pos - i - 1) % buffer.buffer_size
            buffer.rewards[idx] += bonus

        # self.ref_bank.add(tau_new)
        self.episode_count += 1

        if self.use_wandb:
            wandb.log(
                {
                    "preference/signal": s_pref,
                    "preference/bonus_per_step": bonus,
                    "preference/episode_length": T,
                    "timesteps": self.num_timesteps,
                },
                step=self.num_timesteps,
            )


class BestPolicyCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        pref_model,
        best_model,
        ref_bank,
        eval_freq=10000,
        n_eval_episodes=5,
        tolerance=0.02,
        use_wandb=False,
        is_mtl=False,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.pref_model = pref_model
        self.best_model = best_model
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.ref_bank = ref_bank
        self.tolerance = tolerance
        self.use_wandb = use_wandb
        self.is_mtl = is_mtl
        self.eval_round = 0
        self.best_initialized = False
        self.last_best_eval_step = 0

    def _sync_best_model(self):
        self.best_model.set_parameters(self.model.get_parameters(), exact_match=True)

    def _collect_best_bank_trajs(self, model):
        trajs = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            traj = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                traj.append(
                    {
                        "obs": obs,
                        "action": action,
                        "reward": float(reward),
                        "info": info,
                    }
                )

                obs = next_obs

            trajs.append(traj)

        return trajs


    def _collect_trajs(self, model):
        trajs = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            success = 0
            traj = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                traj.append(
                    {
                        "obs": obs,
                        "action": action,
                        "info": info,
                        "reward": reward,
                    }
                )

                success = max(success, int(info.get("success", 0)))
                obs = next_obs

            trajs.append({"traj": traj, "success": success})

        return trajs

    def _compare_sets(self, current_trajs, best_trajs):
        print("COMPARISON:", len(current_trajs), len(best_trajs))
        if not current_trajs or not best_trajs:
            return 0

        directory = f"eval/eval_{self.eval_round}"

        wins = 0
        k = min(len(current_trajs), len(best_trajs))

        for i in range(k):
            winner = self.pref_model.compare(f"{directory}/round_{i}", current_trajs[i]["traj"], best_trajs[i]["traj"])
            if winner == "new":
                wins += 1

        return wins - (k - wins)

    def _on_step(self) -> bool:
        # print(f"[BEST CALLBACK HEARTBEAT] step={self.num_timesteps}")
        if (self.num_timesteps - self.last_best_eval_step) < self.eval_freq:
            return True

        self.last_best_eval_step = self.num_timesteps

        if not self.best_initialized:
            self._sync_best_model()
            best_bank_trajs = self._collect_best_bank_trajs(self.best_model)
            self.ref_bank.replace_all(best_bank_trajs)

            self.best_initialized = True
            self.last_best_eval_step = self.num_timesteps
            return True

        # if self.eval_round == 0:
        #     self._sync_best_model()
        #     self.eval_round += 1
        #     return True

        current_trajs = self._collect_trajs(self.model)
        best_trajs = self._collect_trajs(self.best_model)
        pref_score = self._compare_sets(current_trajs, best_trajs)

        if self.is_mtl:
            success_current = float(np.mean([x["success"] for x in current_trajs]))
            success_best = float(np.mean([x["success"] for x in best_trajs]))
            
            should_update = (
                success_current >= success_best - self.tolerance and pref_score > 0
            )
        else:
            reward_current = float(np.mean([sum(step["reward"] for step in x["traj"]) for x in current_trajs]))
            reward_best = float(np.mean([sum(step["reward"] for step in x["traj"]) for x in best_trajs]))

            should_update = (
                reward_current >= reward_best - self.tolerance and pref_score > 0
            )

        if should_update:
            self._sync_best_model()
            best_bank_trajs = self._collect_best_bank_trajs(self.best_model)
            self.ref_bank.replace_all(best_bank_trajs)

        self.eval_round += 1

        if self.use_wandb:
            if self.is_mtl:
                wandb.log(
                    {
                        "best_policy/success_current": success_current,
                        "best_policy/success_best": success_best,
                        "best_policy/pref_score": pref_score,
                        "best_policy/updated": int(should_update),
                        "timesteps": self.num_timesteps,
                    },
                    step=self.num_timesteps,
                )
            else:
                wandb.log(
                    {
                        "best_policy/reward_current": reward_current,
                        "best_policy/reward_best": reward_best,
                        "best_policy/pref_score": pref_score,
                        "best_policy/updated": int(should_update),
                        "timesteps": self.num_timesteps,
                    },
                    step=self.num_timesteps,
                )

        if self.is_mtl:
            print(
                {
                    "timesteps": self.num_timesteps,
                    "success_current": success_current,
                    "success_best": success_best,
                    "pref_score": pref_score,
                    "best_updated": should_update,
                }
            )
        else:
            print(
                {
                    "timesteps": self.num_timesteps,
                    "reward_current": reward_current,
                    "reward_best": reward_best,
                    "pref_score": pref_score,
                    "best_updated": should_update,
                }
            )

        return True