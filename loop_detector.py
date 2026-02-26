import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------
# 1) Load the pairwise results
# ----------------------------
# Put your filename here (csv with header row exactly like you pasted)
PATH = "/scratch/apoojar4/props-trajectory-wo-env/hopper/overall_log.txt"

df = pd.read_csv(PATH)
print(df.head())
# Basic sanity
required = ["Policy_A_ID", "Policy_B_ID", "Winner"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Candidate set
cands = sorted(set(df["Policy_A_ID"]).union(set(df["Policy_B_ID"])))
n = len(cands)
cand_to_idx = {c: i for i, c in enumerate(cands)}
idx_to_cand = {i: c for c, i in cand_to_idx.items()}

# ----------------------------------------------------------
# 2) Build win matrix W where W[i,j] = times i beats j
#    For your data each pair appears once, so entries are 0/1
# ----------------------------------------------------------
W = np.zeros((n, n), dtype=int)

for _, row in df.iterrows():
    a = int(row["Policy_A_ID"])
    b = int(row["Policy_B_ID"])
    winner = str(row["Winner"]).strip().upper()

    ia, ib = cand_to_idx[a], cand_to_idx[b]

    if winner == "A":
        W[ia, ib] += 1  # A beats B
    elif winner == "B":
        W[ib, ia] += 1  # B beats A
    else:
        raise ValueError(f"Unexpected Winner value: {row['Winner']}")

# Net matrix N where N[i,j] = W[i,j] - W[j,i] (positive => i beats j)
N = W - W.T

# Copeland score = (#wins) - (#losses)
wins = W.sum(axis=1)
losses = W.sum(axis=0)
copeland = wins - losses

# ----------------------------------------------------------
# 3) Sort candidates to make structure/loops easier to see
# ----------------------------------------------------------
order = np.argsort(-copeland)  # descending
W_ord = W[order][:, order]
N_ord = N[order][:, order]
labels_ord = [str(idx_to_cand[i]) for i in order]

# ----------------------------
# 4) Plot matrices
# ----------------------------
def plot_matrix(mat, title, labels, center_zero=False):
    plt.figure(figsize=(8, 7))
    if center_zero:
        vmax = np.max(np.abs(mat))
        vmin = -vmax
    else:
        vmin, vmax = mat.min(), mat.max()

    im = plt.imshow(mat, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=300)

# Win matrix (0/1 in your case)
plot_matrix(W_ord, "Win matrix W (row beats column) — sorted by Copeland", labels_ord, center_zero=False)

# Net matrix (-1/0/1 in your case)
plot_matrix(N_ord, "Net matrix N = W - Wᵀ (positive => row beats column) — sorted by Copeland", labels_ord, center_zero=True)

# -----------------------------------
# 5) Build directed graph + find loops
# -----------------------------------
G = nx.DiGraph()
G.add_nodes_from(cands)

# Add edge i -> j if i beats j at least once
for i in range(n):
    for j in range(n):
        if i != j and W[i, j] > 0:
            u = idx_to_cand[i]
            v = idx_to_cand[j]
            G.add_edge(u, v, weight=int(W[i, j]))

# Find strongly connected components (SCCs). Any SCC size > 1 indicates a loop region.
sccs = list(nx.strongly_connected_components(G))
loop_sccs = [s for s in sccs if len(s) > 1]

print("Candidates:", cands)
print("Copeland scores:", {idx_to_cand[i]: int(copeland[i]) for i in range(n)})
print("\nStrongly connected components (SCCs):")
for s in sccs:
    print(sorted(s))

if loop_sccs:
    print("\nLOOP(S) detected (SCC size > 1):")
    for s in loop_sccs:
        print("  ", sorted(s))
else:
    print("\nNo SCC loops detected (graph is likely acyclic / transitive).")

# Optional: list some explicit directed cycles (can be many; limit output)
cycles = list(nx.simple_cycles(G))
cycles_sorted = sorted(cycles, key=len)
print(f"\nFound {len(cycles_sorted)} directed cycle(s). Showing up to 10 shortest:")
for cyc in cycles_sorted[:10]:
    print("  cycle:", cyc)

# -----------------------------------
# 6) Plot the directed graph
#    Highlight nodes that are in loop SCCs
# -----------------------------------
in_loop = set().union(*loop_sccs) if loop_sccs else set()
node_colors = ["orange" if node in in_loop else "lightgray" for node in G.nodes()]

plt.figure(figsize=(9, 7))

# Layout: spring is usually good for seeing cycles
pos = nx.spring_layout(G, seed=7)

nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors, edgecolors="black", linewidths=1.0)
nx.draw_networkx_labels(G, pos, font_size=10)

# Draw edges with arrows
nx.draw_networkx_edges(
    G, pos,
    arrowstyle="-|>",
    arrowsize=16,
    width=1.5,
    connectionstyle="arc3,rad=0.08"  # curved edges help visibility
)

plt.title("Pairwise winners as a directed graph (orange nodes participate in a loop SCC)")
plt.axis("off")
plt.tight_layout()

plt.savefig("pairwise_graph.png", dpi=300)