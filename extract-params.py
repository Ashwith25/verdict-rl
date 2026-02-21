import os
import random


for i in [3, 4, 5]:
    file = f"/scratch/apoojar4/props-log/mcd-logs/MCD_traj_20_summary/trial_{i}"
    reward_file_lines = open(f"/scratch/apoojar4/props-log/mcd-logs/MCD_traj_20_summary/trial_{i}/overall_log.txt").readlines()[1:]

    with open("mcd_dataset.txt", "a") as param_file:
        filenames = [f for f in os.listdir(file) if f.startswith("episode")]
        sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1]))
        for filename in sorted_filenames:
            # print(filename)
            with open(os.path.join(file, filename, "training_rollout.txt"), "r") as f:
                lines = f.readlines()
                param_file.write(f"{lines[0].strip()} | {reward_file_lines[int(filename.split('_')[-1])].split(',')[-2].strip()}\n")

# The above code extracts parameters and their corresponding rewards from log files and writes them to a new file.
# Want to categorize based on reward ranges? 0-25%, 25-50%, 50-75%, 75-100%
# with open("ip-params-categorized.txt", "w") as categorized_file:
#     rewards = []
#     with open("ip-params.txt", "r") as param_file:
#         for line in param_file:
#             parts = line.strip().split(" | ")
#             if len(parts) == 2:
#                 param, reward_str = parts
#                 try:
#                     reward = float(reward_str)
#                     rewards.append((param, reward))
#                 except ValueError:
#                     continue

#     if not rewards:
#         print("No valid rewards found.")
#         exit(1)

#     min_reward = min(r[1] for r in rewards)
#     max_reward = max(r[1] for r in rewards)
#     range_size = (max_reward - min_reward) / 4

#     categories = {'1': [], '2': [], '3': [], '4': []}
#     for param, reward in rewards:
#         if reward <= min_reward + range_size:
#             category = "1"
#         elif reward <= min_reward + 2 * range_size:
#             category = "2"
#         elif reward <= min_reward + 3 * range_size:
#             category = "3"
#         else:
#             category = "4"
#         categorized_file.write(f"{param} | {reward} | {category}\n")
#         categories[category].append((param, reward))

# # Sample 100 params from each category
# sampled_params = []
# samples_per_category = 105

# for cat, items in categories.items():
#     if len(items) >= samples_per_category:
#         sampled_params.extend(random.sample(items, samples_per_category))
#     else:
#         print(f"Warning: Category {cat} has only {len(items)} items, taking all.")
#         sampled_params.extend(items)

# with open("ip-params-sampled.txt", "w") as sampled_file:
#     for param, reward in sampled_params:
#         sampled_file.write(f"{param} | {reward}\n")


