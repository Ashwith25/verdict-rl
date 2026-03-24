import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import spearmanr
import os

# --- CONFIGURATION ---
LOG_FILE = "props-rlm/hopper/overall_log.txt"
OUTPUT_DIR = "traj-plots-MLE/hopper-30"
OUTPUT_CSV = "traj-plots-MLE/hopper-30/ranking_comparison.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

loss_history = []

def load_and_optimize(filepath):
    """
    Re-runs the MLE optimization to get the scores (Same as before).
    """
    matches = []
    unique_ids = set()
    true_rewards = {} 
    
    if not os.path.exists(filepath): raise FileNotFoundError(f"{filepath} not found")

    with open(filepath, 'r') as f:
        f.readline()
        for line in f:
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 6: continue
                id_a, id_b = int(parts[1]), int(parts[2])
                true_rewards[id_a] = float(parts[3])
                true_rewards[id_b] = float(parts[4])
                unique_ids.add(id_a); unique_ids.add(id_b)
                
                winner = parts[5].upper()
                score_a = 1.0 if winner == "A" else 0.0 if winner == "B" else 0.5
                matches.append((id_a, id_b, score_a))
            except: continue

    dense_to_sparse = sorted(list(unique_ids))
    sparse_to_dense = {original: i for i, original in enumerate(dense_to_sparse)}
    
    dense_matches = [(sparse_to_dense[a], sparse_to_dense[b], s) for a, b, s in matches]
    num_policies = len(unique_ids)

    # MLE Optimization
    def nll(params):
        loss = 0
        for (a, b, s) in dense_matches:
            prob = 1 / (1 + np.exp(params[b] - params[a]))
            loss -= (s * np.log(prob + 1e-9) + (1 - s) * np.log(1 - prob + 1e-9))
        return loss + 0.01 * np.sum(params**2)

    res = minimize(nll, np.zeros(num_policies), method='BFGS')
    learned_scores = res.x
    
    return learned_scores, dense_to_sparse, true_rewards

def main():
    
    learned_scores, id_map, true_rewards_map = load_and_optimize(LOG_FILE)
    num_policies = len(learned_scores)

    # 1. Prepare Ranking Data
    # We create a list of dicts to easily sort
    data = []
    for i in range(num_policies):
        orig_id = id_map[i]
        data.append({
            "id": orig_id,
            "true_reward": true_rewards_map[orig_id],
            "learned_score": learned_scores[i]
        })

    # Sort by True Reward (Descending) -> Assign True Rank
    data.sort(key=lambda x: x["true_reward"], reverse=True)
    for rank, item in enumerate(data):
        item["true_rank"] = rank + 1 # 1-based ranking

    # Sort by Learned Score (Descending) -> Assign Learned Rank
    data.sort(key=lambda x: x["learned_score"], reverse=True)
    for rank, item in enumerate(data):
        item["learned_rank"] = rank + 1

    # Extract lists for plotting
    true_ranks = [d["true_rank"] for d in data]
    learned_ranks = [d["learned_rank"] for d in data]
    ids = [d["id"] for d in data]

    # --- PLOT 1: Rank vs Rank Scatter ---
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(x=true_ranks, y=learned_ranks, s=150, color="#8e44ad", edgecolor="k")
    
    # Draw Perfect Alignment Line
    max_rank = max(max(true_ranks), max(learned_ranks))
    plt.plot([0, max_rank+1], [0, max_rank+1], '--', color="gray", label="Perfect Alignment")
    
    # Add labels for outliers
    for i, txt in enumerate(ids):
        # Annotate if rank difference is significant (>2 spots)
        if abs(true_ranks[i] - learned_ranks[i]) > 2:
            plt.annotate(f"ID {txt}", (true_ranks[i], learned_ranks[i]), xytext=(5,5), textcoords='offset points')

    # Calculate Spearman Correlation (Ordinal Correlation)
    rho, p_val = spearmanr(true_ranks, learned_ranks)

    plt.title(f"Rank Alignment Check\nSpearman Rho = {rho:.4f}", fontsize=16, fontweight='bold')
    plt.xlabel("True Rank (Physics Engine)", fontsize=14)
    plt.ylabel("Predicted Rank (LLM)", fontsize=14)
    plt.gca().invert_xaxis() # Rank 1 is usually top-right or top-left, let's keep 1 at bottom-left for cartesian logic
    plt.gca().invert_yaxis() # Rank 1 at top/right
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rank_vs_rank.png", dpi=300)

    # --- PLOT 2: Truth-Ordered Heatmap ---
    # We construct the matrix based on TRUE REWARD sorting
    # If the model is good, this should still look smooth.
    
    # 1. Get indices sorted by True Reward (Best to Worst)
    # We need to map back to the 'learned_scores' array indices
    truth_sorted_indices = []
    # Re-sort data by true reward to get the order
    data.sort(key=lambda x: x["true_reward"], reverse=True)
    
    # Find the original index for each item in the sorted list
    # This is inefficient but clear:
    for item in data:
        # Find index in id_map
        idx = id_map.index(item["id"])
        truth_sorted_indices.append(idx)

    prob_matrix = np.zeros((num_policies, num_policies))
    
    for r, i in enumerate(truth_sorted_indices):     # Row = Policy i
        for c, j in enumerate(truth_sorted_indices): # Col = Policy j
            score_i = learned_scores[i]
            score_j = learned_scores[j]
            prob_matrix[r, c] = 1 / (1 + np.exp(score_j - score_i))

    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_matrix, cmap="RdBu_r", vmin=0, vmax=1, center=0.5)
    
    plt.title("Win Probability Matrix\n(Sorted by TRUE PHYSICS REWARD)", fontsize=16, fontweight='bold')
    plt.xlabel("Opponent Rank (Physics)", fontsize=12)
    plt.ylabel("Policy Rank (Physics)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/truth_sorted_heatmap.png", dpi=300)

    print(f"Spearman Correlation (Ranking Consistency): {rho:.4f}")
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()