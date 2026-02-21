import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os
import argparse


# --- CONFIGURATION ---
# LOG_FILE = "traj-logs/hopper/overall_log.txt"
# OUTPUT_DIR = "traj-plots-MLE/hopper-10"
# OUTPUT_CSV = "traj-plots-MLE/hopper-10/ranking_comparison.csv"
# LOG_FILE = "props-BT-2/hopper/trial_1/overall_log.txt"



loss_history = []

def load_data(filepath):
    """
    Parses the tournament log.
    Expected Header: Round_ID | Policy_A_ID | Policy_B_ID | True_Reward_A | True_Reward_B | Winner
    """
    matches = []       # List of (ID_A, ID_B, score_A)
    unique_ids = set()
    true_rewards = {}  # Map ID -> Ground Truth Reward
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'r') as f:
        header = f.readline() # Skip header
        for line in f:
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 6: continue
                
                id_a = int(parts[1])
                id_b = int(parts[2])
                true_rewards[id_a] = true_rewards.get(id_a, []) + [float(parts[3])]
                true_rewards[id_b] = true_rewards.get(id_b, []) + [float(parts[4])]
                unique_ids.add(id_a)
                unique_ids.add(id_b)
                
                winner = parts[5].upper()
                score_a = 1.0 if winner == "A" else 0.0 if winner == "B" else 0.5
                score_true = 1.0 if float(parts[3]) > float(parts[4]) else 0.0 if float(parts[4]) > float(parts[3]) else 0.5
                matches.append((id_a, id_b, score_a, score_true))
            except ValueError:
                continue

    # Convert sparse IDs to dense indices (0..N-1)
    dense_to_sparse = sorted(list(unique_ids))
    sparse_to_dense = {original: i for i, original in enumerate(dense_to_sparse)}
    
    dense_matches = []
    for (a, b, score_a, score_true) in matches:
        dense_matches.append((sparse_to_dense[a], sparse_to_dense[b], score_a, score_true))
        
    print(f"Loaded {len(matches)} matches involving {len(unique_ids)} unique policies.")

    # Average true rewards if multiple entries exist
    for pid in true_rewards:
        true_rewards[pid] = np.mean(true_rewards[pid])

    return dense_matches, dense_to_sparse, true_rewards

def negative_log_likelihood(params, matches):
    """
    Minimizes NLL to find the best-fit Bradley-Terry scores.
    """
    loss = 0
    epsilon = 1e-9 
    
    for (idx_a, idx_b, score_a, _) in matches:
        r_a = params[idx_a]
        r_b = params[idx_b]
        # P(A wins) = sigmoid(r_a - r_b)
        prob_a_wins = 1 / (1 + np.exp(r_b - r_a))
        loss -= (score_a * np.log(prob_a_wins + epsilon) + 
                 (1 - score_a) * np.log(1 - prob_a_wins + epsilon))
        
    loss += 0.01 * np.sum(params**2) # L2 Regularization
    return loss

def negative_log_likelihood_true(params, matches):
    """
    Minimizes NLL to find the best-fit Bradley-Terry scores.
    """
    loss = 0
    epsilon = 1e-9 
    
    for (idx_a, idx_b, _ , score_true) in matches:
        r_a = params[idx_a]
        r_b = params[idx_b]
        # P(A wins) = sigmoid(r_a - r_b)
        prob_a_wins = 1 / (1 + np.exp(r_b - r_a))
        loss -= (score_true * np.log(prob_a_wins + epsilon) + 
                 (1 - score_true) * np.log(1 - prob_a_wins + epsilon))
        
    loss += 0.01 * np.sum(params**2) # L2 Regularization
    return loss

def main(logdir=None):
    # ... inside main() ...
    
    # Reset history (in case you run main multiple times)

    OUTPUT_DIR = f"{logdir}/mle_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_CSV = f"{OUTPUT_DIR}/ranking_comparison.csv"
    OUTPUT_TXT = f"{OUTPUT_DIR}/ranking_comparison.txt"

    loss_history.clear()

    # Define a wrapper that saves the loss before returning it
    def objective_with_tracking(params):
        val = negative_log_likelihood(params, matches)
        loss_history.append(val)
        return val
    # 1. Load & Optimize
    matches, id_map, true_rewards_map = load_data(f"{logdir}/overall_log.txt")
    if not matches: return

    num_policies = len(id_map)
    print(f"Running MLE Optimization on {num_policies} policies...")
    
    result = minimize(
        objective_with_tracking,
        np.zeros(num_policies),
        # args=(matches,),
        method='BFGS'
    )

    result_true = minimize(
        negative_log_likelihood_true,
        np.zeros(num_policies),
        args=(matches,),
        method='BFGS'
    )
    learned_scores = result.x
    learned_true_scores = result_true.x
    # print("MATCHES:", matches)
    # print("LEARNED SCORES:", learned_scores)
    # print("LEARNED TRUE SCORES:", learned_true_scores)
    print("Optimization Complete.")

    # Plot the Training Curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Negative Log Likelihood", color="red")
    plt.xlabel("Optimizer Steps")
    plt.ylabel("Loss")
    plt.title("MLE Training Curve (Did it learn?)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/training_curve.png")
    print(f"Saved training curve to {OUTPUT_DIR}/training_curve.png")

    # 2. Extract Results
    y_true = []
    y_pred = []
    # print("\n--- Final Results (Sample) ---")
    # print(f"{'Policy ID':<10} | {'True Reward':<15} | {'Learned Score':<20}")
    # print("-" * 55)
    
    sorted_indices = np.argsort([true_rewards_map[orig] for orig in id_map])
    for i in sorted_indices:
        orig_id = id_map[i]
        # y_true.append(true_rewards_map[orig_id])
        y_true.append(learned_true_scores[i])
        y_pred.append(learned_scores[i])
        # if i % (max(1, num_policies//10)) == 0:
        # print(f"{orig_id:<10} | {true_rewards_map[orig_id]:<15.2f} | {learned_scores[i]:<20.4f}")

    # 3. Plot 1: Correlation Scatter
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x=y_true, y=y_pred, s=100, color="#2980b9", edgecolor="k")
    sns.regplot(x=y_true, y=y_pred, scatter=False, color="#c0392b")
    
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.title(f"Bradley-Terry Validation (R = {corr:.4f})", fontsize=16, fontweight='bold')
    plt.xlabel("Ground Truth Reward (Physics)", fontsize=14)
    plt.ylabel("Learned Preference Score (LLM)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bradley_terry_correlation.png"), dpi=300)

    # 4. Plot 2: Win Probability Matrix (Heatmap)
    # Sort policies from Best (High Score) to Worst (Low Score)
    rank_indices = np.argsort(learned_scores)[::-1] # Reverse sort (Best first)
    n = len(rank_indices)
    prob_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            score_i = learned_scores[rank_indices[i]]
            score_j = learned_scores[rank_indices[j]]
            # Probability i beats j
            prob_matrix[i, j] = 1 / (1 + np.exp(score_j - score_i))

    plt.figure(figsize=(10, 8))
    # Red = High Prob (Win), Blue = Low Prob (Loss)
    sns.heatmap(prob_matrix, cmap="RdBu_r", vmin=0, vmax=1, center=0.5, square=True)
    
    plt.title("Predicted Win Probability Matrix\n(Sorted: Best Policy at Top-Left)", fontsize=16, fontweight='bold')
    plt.xlabel("Opponent Policy Rank (Left=Best, Right=Worst)", fontsize=12)
    plt.ylabel("Policy Rank (Top=Best, Bottom=Worst)", fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, "win_probability_matrix.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"\nPlots saved to {OUTPUT_DIR}/")

    # 1. Build Data List
    data = []
    for i, score in enumerate(learned_scores):
        pid = id_map[i]
        data.append({
            "Policy_ID": pid,
            "True_Reward": true_rewards_map[pid],
            "LLM_Score": score,
            "MLE_True_Score": learned_true_scores[i]
        })
    
    # 2. Calculate True Ranks (Sort by Reward Descending)
    data.sort(key=lambda x: x["True_Reward"], reverse=True)
    for rank, item in enumerate(data):
        item["True_Rank"] = rank + 1

    # 3. Calculate LLM Ranks (Sort by Score Descending)
    data.sort(key=lambda x: x["LLM_Score"], reverse=True)
    for rank, item in enumerate(data):
        item["LLM_Rank"] = rank + 1

    # 4. Calculate Delta and Final Sort
    # We usually want to view the table sorted by True Rank to see "Best to Worst"
    for item in data:
        item["Rank_Delta"] = item["True_Rank"] - item["LLM_Rank"]
        # Positive Delta = True(5) - LLM(1) = +4 (LLM Overrated it)
        # Negative Delta = True(1) - LLM(5) = -4 (LLM Underrated it)

    data.sort(key=lambda x: x["True_Rank"]) # Sort by Physics Rank for the table

    with open(OUTPUT_TXT, 'w') as f:
        f.write("="*85 + "\n")
        f.write(f"{'ID':<5} | {'True Reward':<12} | {'LLM Score':<10} | {'MLE True Score':<15} | {'Phys Rank':<10} | {'LLM Rank':<10} | {'Delta':<6}\n")
        f.write("="*85 + "\n")

        # # 5. Print Table
        # print("\n" + "="*85)
        # print(f"{'ID':<5} | {'True Reward':<12} | {'LLM Score':<10} | {'MLE True Score':<15} | {'Phys Rank':<10} | {'LLM Rank':<10} | {'Delta':<6}")
        # print("="*85)

        for row in data:
            # Highlight significant disagreements
            delta_str = f"{row['Rank_Delta']:+d}"
            if abs(row['Rank_Delta']) >= 5:
                delta_str += " ⚠️" # Major disagreement
            elif row['Rank_Delta'] == 0:
                delta_str = " ✔️"  # Perfect match
                
            f.write(f"{row['Policy_ID']:<5} | {row['True_Reward']:<12.2f} | {row['LLM_Score']:<10.4f} | {row['MLE_True_Score']:<15.4f} | "
                    f"{row['True_Rank']:<10} | {row['LLM_Rank']:<10} | {delta_str:<6}\n")

        f.write("="*85 + "\n")
        f.write("Interpretation:\n")
        f.write("  Phys Rank: 1 is Best (Physics).\n")
        f.write("  LLM Rank:  1 is Best (LLM).\n")
        f.write("  Delta > 0: LLM ranked it HIGHER/BETTER than Physics (Overrated).\n")
        f.write("  Delta < 0: LLM ranked it LOWER/WORSE than Physics (Underrated/Safety Penalty).\n")

        # Save to CSV for PPT
        df = pd.DataFrame(data)
        df = df[["Policy_ID", "True_Reward", "LLM_Score", "True_Rank", "LLM_Rank", "Rank_Delta"]]
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved detailed table to {OUTPUT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional override for logdir from the config file",
    )
    args = parser.parse_args()
    
    for folders in os.listdir(args.log_dir):
        folder_path = os.path.join(args.log_dir, folders)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "overall_log.txt")):
            print(f"\nProcessing folder: {folder_path}")
            main(logdir=folder_path)
        else:
            print(f"Skipping {folder_path} (not a directory or missing overall_log.txt)")