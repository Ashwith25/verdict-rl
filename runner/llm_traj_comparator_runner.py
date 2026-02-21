from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from agent.trajectory_agent import LLMNumOptimTrajectoryAgent
from agent.nn_num_optim_reward import NNNumOptimRewardAgent
from jinja2 import Environment, FileSystemLoader
import os
import traceback
import numpy as np
import random

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
    print(os.environ.get('OLLAMA_HOST'))
    print(os.environ)

    assert dataset_file is not None, "Dataset file for parameters must be provided."
    with open(dataset_file, "r") as f:
        params_dataset = [np.array(l.split(" | ")[0].split(",")).astype(float) for l in f.readlines()]

    final_samples = random.sample(params_dataset, 10)
    final_comparison_policies = [[i, j, final_samples[i], final_samples[j]] for i in range(len(final_samples)) for j in range(i+1, len(final_samples))]
    print("length of final comparison policies:", len(final_comparison_policies))
    
    overall_log_file = open(f"{logdir}/overall_log.txt", "w")
    overall_log_file.write("Iteration, Policy_A_ID, Policy_B_ID, True_Reward_A, True_Reward_B, Winner\n")
    overall_log_file.flush()
    for episode, (policy_A_ID, policy_B_ID, policy_A, policy_B) in enumerate(final_comparison_policies):
        print(f"Comparison: {episode}")
        # create log dir
        curr_episode_dir = f"{logdir}/comparison_{episode}"
        print(f"Creating log directory: {curr_episode_dir}")
        os.makedirs(curr_episode_dir, exist_ok=True)
        
        for trial_idx in range(5):
            try:
                predicted_winner, true_reward = agent.train_policy(world, curr_episode_dir, policy_A, policy_B)
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
