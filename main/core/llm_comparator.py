import numpy as np
import json
import sys
import os

# Add parent directory to path to enable imports from agent module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent.policy.llm_brain_trajectory import LLMBrainTrajectory
from jinja2 import Environment, FileSystemLoader

class LLMComparator:
    def __init__(
            self,
            template_dir,
            llm_si_template_name,
            llm_output_conversion_template_name,
            llm_model_name,
            task_json_path,
            logdir = "./verdict-rl-logs",
            use_llm_pref=False,
            **kwargs
        ):
        self.use_llm_pref = use_llm_pref
        self.logdir = logdir
        task_json = json.load(open(task_json_path, "r")) if task_json_path is not None else "N/A"
        goal_description = task_json.get("goal_description", "N/A") if isinstance(task_json, dict) else "N/A"
        state_description = task_json.get("state_description", "N/A") if isinstance(task_json, dict) else "N/A"
        task_description = task_json.get("task_description", "N/A") if isinstance(task_json, dict) else "N/A"

        self.context = {
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
        jinja2_env = Environment(loader=FileSystemLoader(template_dir))
        llm_si_template = jinja2_env.get_template(llm_si_template_name)
        llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
        self.llm_traj_brain = LLMBrainTrajectory(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )

    def compare(self, directory, tau_new, tau_ref):
        if not self.use_llm_pref:
            return "ref"
        
        # print("Comparing trajectories:")
        # print("New trajectory:", len(tau_new))
        # print("Reference trajectory:", len(tau_ref))

        # print("Using LLM to compare trajectories...")
        # print("Trajectory 1:", tau_new)
        # print("Trajectory 2:", tau_ref)

        curr_episode_dir = f"{self.logdir}/{directory}"
        os.makedirs(curr_episode_dir, exist_ok=True)

        winner = self.llm_traj_brain.llm_update_parameters_num_optim_semantics(
            context=self.context,
            trajectory_a=tau_new,
            trajectory_b=tau_ref,
            log_dir=curr_episode_dir
        )

        return "ref" if winner == "B" else "new"