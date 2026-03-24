from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from agent.trajectory_agent import LLMNumOptimTrajectoryAgent
from agent.nn_num_optim_reward import NNNumOptimRewardAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np
import random
import json

def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    task_json,
    traj_dir_path,
    summary_template = None,
    summary_desc_file = None,
    bias=None,
    rank=None,
    optimum=1000,
    search_step_size=0.1,
    env_kwargs=None,
    dataset_file=None,
    reward_range=None,
    env_desc_file=None,
    run_nn_train=False,
):
    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )

    world = ContinualSpaceGeneralWorld(
        gym_env_name,
        render_mode,
        max_traj_length,
    )

    agent = LLMNumOptimTrajectoryAgent(
        logdir,
        dim_actions,
        dim_states,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        bias,
        env_desc_file=env_desc_file,
    )

    print('init done')
    # print(os.environ.get('OLLAMA_HOST'))
    # print(os.environ)

    # assert dataset_file is not None, "Dataset file for parameters must be provided."

    if traj_dir_path is None:
        with open(dataset_file, "r") as f:
            params_dataset = [np.array(l.split(" | ")[0].split(",")).astype(float) for l in f.readlines()]

        final_samples = random.sample(params_dataset, 5)
        final_comparison_policies = [[i, j, final_samples[i], final_samples[j]] for i in range(len(final_samples)) for j in range(i+1, len(final_samples))]
        print("length of final comparison policies:", len(final_comparison_policies))
    else:
        policies_path = ["rollouts_2eps/trajectories/reach-v3__ppo__seed0__ep000.npz", "rollouts_2eps/trajectories/reach-v3__ppo__seed1__ep000.npz", "rollouts_2eps/trajectories/reach-v3__sac__seed0__ep000.npz", "rollouts_2eps/trajectories/reach-v3__sac__seed3__ep001.npz", "rollouts_2eps/trajectories/reach-v3__td3__seed1__ep001.npz"]
        final_comparison_policies = [[i, j, policies_path[i], policies_path[j]] for i in range(len(policies_path)) for j in range(i+1, len(policies_path))]

    overall_log_file = open(f"{logdir}/overall_log.txt", "w")
    overall_log_file.write("Iteration, Policy_A_ID, Policy_B_ID, True_Reward_A, True_Reward_B, Winner\n")
    overall_log_file.flush()

    task_json = json.load(open(task_json, "r")) if task_json is not None else "N/A"
    goal_description = task_json.get("goal_description", "N/A") if isinstance(task_json, dict) else "N/A"
    state_description = task_json.get("state_description", "N/A") if isinstance(task_json, dict) else "N/A"
    task_description = task_json.get("task_description", "N/A") if isinstance(task_json, dict) else "N/A"

    context = {
        "task_description": task_description,
        "state_action_meaning": state_description,
        "goal": goal_description,
        "evaluation_criteria": [
            "goal attainment",
            "stability near goal",
            "smoothness",
            "safety",
            "efficiency"
        ]
    }

    # print(final_comparison_policies)

    if traj_dir_path is not None:
        for episode, (policy_A_ID, policy_B_ID, traj_A_path, traj_B_path) in enumerate(final_comparison_policies):
            print(f"Comparison: {episode}")
            # create log dir
            curr_episode_dir = f"{logdir}/comparison_{episode}"
            print(f"Creating log directory: {curr_episode_dir}")
            os.makedirs(curr_episode_dir, exist_ok=True)
            
            for trial_idx in range(5):
                try:
                    predicted_winner, true_reward = agent.train_policy(world, curr_episode_dir, traj_A_path, traj_B_path, context, traj_A_path=traj_A_path, traj_B_path=traj_B_path)
                    overall_log_file.write(f"{episode + 1}, {policy_A_ID}, {policy_B_ID}, {true_reward[0]}, {true_reward[1]}, {predicted_winner}\n")
                    overall_log_file.flush()
                    print(f"{trial_idx + 1}th trial attempt succeeded in training")
                    break
                except Exception as e:
                    print(
                        f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                    )
                    traceback.print_exc()
                    continue
            if trial_idx == 4:
                print(f"Episode {episode} failed to train after 5 attempts")
                break
    else:
        for episode, (policy_A_ID, policy_B_ID, policy_A, policy_B) in enumerate(final_comparison_policies):
            print(f"Comparison: {episode}")
            # create log dir
            curr_episode_dir = f"{logdir}/comparison_{episode}"
            print(f"Creating log directory: {curr_episode_dir}")
            os.makedirs(curr_episode_dir, exist_ok=True)
            
            for trial_idx in range(5):
                try:
                    predicted_winner, true_reward = agent.train_policy(world, curr_episode_dir, policy_A, policy_B, context, traj_A_path=traj_A_path, traj_B_path=traj_B_path)
                    overall_log_file.write(f"{episode + 1}, {policy_A_ID}, {policy_B_ID}, {true_reward[0]}, {true_reward[1]}, {predicted_winner}\n")
                    overall_log_file.flush()
                    print(f"{trial_idx + 1}th trial attempt succeeded in training")
                    break
                except Exception as e:
                    print(
                        f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                    )
                    traceback.print_exc()
                    continue
            if trial_idx == 4:
                print(f"Episode {episode} failed to train after 5 attempts")
                break
    overall_log_file.close()
