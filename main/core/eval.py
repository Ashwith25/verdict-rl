import numpy as np


def evaluate(model, env, n_episodes=10):
    rewards = []
    successes = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        success = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            success = max(success, int(info.get("success", 0)))

        rewards.append(total_reward)
        successes.append(success)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
    }