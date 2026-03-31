# VERDICT-RL
### Learning from Comparisons: LLMs as Critics in Reinforcement Learning

Replace fragile reward signals with reasoned judgment.

VERDICT-RL uses Large Language Models as critics to guide reinforcement learning via trajectory comparisons instead of scalar rewards.

---

## Overview

Traditional RL optimizes a scalar reward.  
This breaks down when:
- rewards are poorly specified  
- policies exploit loopholes (reward hacking)  
- tasks require semantic understanding  

VERDICT-RL replaces rewards with comparisons.

Instead of asking:
“How good is this trajectory?”

We ask:
“Which trajectory is better?”

---

## Training Loop

```mermaid
sequenceDiagram
    participant SAC as Agent
    participant Env as Environment
    participant LLM as LLM Preference Model
    participant Bank as Best Trajectory Buffer
    participant Best as Best Policy

    %% ===== Training Loop =====
    rect rgb(196,122,83)
        note over SAC,LLM: Per-Episode Training Loop

        SAC->>Env: Rollout trajectory τ_new
        Env-->>SAC: states, actions, rewards

        SAC->>Bank: Sample τ_best
        Bank-->>SAC: τ_best

        SAC->>LLM: Compare τ_new vs τ_best
        LLM-->>SAC: s_pref ∈ {+1, -1}

        SAC->>SAC: Modify reward\nr = r_env + (α * s_pref / T)
    end

    %% ===== Best Policy Update =====
    rect rgb(143,71,49)
        note over SAC,Best: Periodic Best-Policy Update

        SAC->>Env: Rollout N trajectories (current)
        Best->>Env: Rollout N trajectories (best)

        SAC->>LLM: Compare trajectory sets
        LLM-->>SAC: pref_score

        alt If current is better AND preferred
            SAC->>Best: Update π_best ← π
            Best->>Bank: Refresh τ_best set
        end
    end
```

---

## Key Idea

VERDICT-RL shifts learning from:

| Traditional RL | VERDICT-RL |
|----------------|-----------|
| Scalar reward | Pairwise preference |
| Hand-designed signals | LLM reasoning |
| Opaque optimization | Interpretable critique |

---

## Setup

```bash
git clone https://github.com/your-username/verdict-rl.git
cd verdict-rl

conda create -n verdict python=3.10
conda activate verdict

pip install -r requirements.txt
```

---

## Training

Baseline (SAC)

```bash
python train_sac.py --env reach-v3 --wandb --run_name sac-reach-v3-seed0
```

## VERDICT-RL

```bash
python train_verdict.py --env reach-v3 --use_llm_pref --wandb --run_name verdict-reach-v3-seed0
```
---

Run the two experiments separately for a clean comparison

## Logging

Login to Weights & Biases once before training:
```bash
wandb login
```

## Optional but worth adding

If vanilla SAC should stay on CPU because your LLM eats GPU memory, add:

```bash
python train_sac.py --env reach-v3 --wandb --device cpu
```

---

## Example Critic Prompt

```
You are an expert RL evaluator.

Compare two trajectories and decide which better achieves the goal.

Return:
1. Preferred trajectory (A or B)
2. Short reasoning
```

---

## Limitations

- LLM inference cost  
- Prompt sensitivity  
- Bias in comparisons  
- Training latency  

---

## Takeaway

Rewards compress behavior into a number.  
Comparisons preserve meaning.

VERDICT-RL learns from judgment, not just signals.
