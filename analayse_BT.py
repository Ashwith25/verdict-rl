import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os

# --- CONFIGURATION ---
LOG_FILE = "traj-logs/hopper/overall_log.txt"
OUTPUT_DIR = "traj-plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                true_rewards[id_a] = float(parts[3])
                true_rewards[id_b] = float(parts[4])
                unique_ids.add(id_a)
                unique_ids.add(id_b)
                
                winner = parts[5].upper()
                score_a = 1.0 if winner == "A" else 0.0 if winner == "B" else 0.5
                matches.append((id_a, id_b, score_a))
            except ValueError:
                continue

    # Convert sparse IDs to dense indices (0..N-1)
    dense_to_sparse = sorted(list(unique_ids))
    sparse_to_dense = {original: i for i, original in enumerate(dense_to_sparse)}
    
    dense_matches = []
    for (a, b, score) in matches:
        dense_matches.append((sparse_to_dense[a], sparse_to_dense[b], score))
        
    print(f"Loaded {len(matches)} matches involving {len(unique_ids)} unique policies.")
    return dense_matches, dense_to_sparse, true_rewards

def negative_log_likelihood(params, matches):
    """
    Minimizes NLL to find the best-fit Bradley-Terry scores.
    """
    loss = 0
    epsilon = 1e-9 
    
    for (idx_a, idx_b, score_a) in matches:
        r_a = params[idx_a]
        r_b = params[idx_b]
        # P(A wins) = sigmoid(r_a - r_b)
        prob_a_wins = 1 / (1 + np.exp(r_b - r_a))
        loss -= (score_a * np.log(prob_a_wins + epsilon) + 
                 (1 - score_a) * np.log(1 - prob_a_wins + epsilon))
        
    loss += 0.01 * np.sum(params**2) # L2 Regularization
    return loss

def main():
    # 1. Load & Optimize
    matches, id_map, true_rewards_map = load_data(LOG_FILE)
    if not matches: return

    num_policies = len(id_map)
    print(f"Running MLE Optimization on {num_policies} policies...")
    
    result = minimize(
        negative_log_likelihood,
        np.zeros(num_policies),
        args=(matches,),
        method='BFGS'
    )
    learned_scores = result.x
    print("Optimization Complete.")

    # 2. Extract Results
    y_true = []
    y_pred = []
    print("\n--- Final Results (Sample) ---")
    print(f"{'Policy ID':<10} | {'True Reward':<15} | {'Learned Score':<20}")
    print("-" * 55)
    
    sorted_indices = np.argsort([true_rewards_map[orig] for orig in id_map])
    for i in sorted_indices:
        orig_id = id_map[i]
        y_true.append(true_rewards_map[orig_id])
        y_pred.append(learned_scores[i])
        if i % (max(1, num_policies//10)) == 0:
            print(f"{orig_id:<10} | {true_rewards_map[orig_id]:<15.2f} | {learned_scores[i]:<20.4f}")

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

if __name__ == "__main__":
    main()