import numpy as np
import gymnasium as gym

def get_action(dim_states, dim_actions, state, params):
    # print(params.shape, dim_states, dim_actions, state.shape)
    W = params[:(dim_actions * dim_states)].reshape(dim_states, dim_actions)
    b = params[dim_actions * dim_states:dim_actions * dim_states + dim_actions]

    action = state @ W + b
    return np.clip(action, -1.0, 1.0)

def rollout(env, dim_states, dim_actions, params, max_steps=10000):
    obs, _ = env.reset()
    
    states, actions = [], []
    total_reward = 0

    trajectory = []

    for i in range(max_steps):
        action = get_action(dim_states, dim_actions, obs, params)

        # states.append(obs.copy())
        # actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        trajectory.append(
            {
                "obs": np.array(obs),
                "action": np.array(action),
                "reward": float(reward),
                "info": info,
                "timestep": i,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        if terminated or truncated:
            break

    return trajectory