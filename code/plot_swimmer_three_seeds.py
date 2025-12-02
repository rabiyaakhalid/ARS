import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# paths for your logs
v1_files = [
    "results/V1/Swimmer/Iteration1/seed1.txt",
    "results/V1/Swimmer/Iteration2/seed2.txt",
    "results/V1/Swimmer/Iteration3/seed3.txt",
]

v2_files = [
    "results/V2/Swimmer/Iteration1/seed1.txt",
    "results/V2/Swimmer/Iteration2/seed2.txt",
    "results/V2/Swimmer/Iteration3/seed3.txt",
]

def load_run(path, max_iters=1000):
    """Return episodes array and reward array from one seed file."""
    df = pd.read_csv(path, sep="\t")
    # keep only first max_iters rows in case you run longer later
    df = df.iloc[:max_iters]
    episodes = df["Iteration"].to_numpy()
    rewards = df["AverageReward"].to_numpy()
    return episodes, rewards

def stack_runs(file_list, max_iters=1000):
    """Load several seeds and stack rewards into a 2D array."""
    all_rewards = []
    episodes = None
    for p in file_list:
        eps, r = load_run(p, max_iters=max_iters)
        if episodes is None:
            episodes = eps
        all_rewards.append(r)
    return episodes, np.vstack(all_rewards)

# load V1 and V2
episodes_v1, R_v1 = stack_runs(v1_files, max_iters=1000)
episodes_v2, R_v2 = stack_runs(v2_files, max_iters=1000)

# statistics across seeds
v1_mean = R_v1.mean(axis=0)
v1_std = R_v1.std(axis=0)

v2_mean = R_v2.mean(axis=0)
v2_std = R_v2.std(axis=0)

episodes = episodes_v1  # same for all runs

# paper like colors
V1_LINE = "#4c72b0"   # blue line
V1_SHADE = "#aec0dc"  # blue shade

V2_LINE = "#55a868"   # green line
V2_SHADE = "#b7d8c4"  # green shade

plt.figure(figsize=(6,4))

# ARS V1
plt.plot(episodes, v1_mean, color=V1_LINE, linewidth=2, label="ARS V1")
plt.fill_between(
    episodes,
    v1_mean - v1_std,
    v1_mean + v1_std,
    color=V1_SHADE,
    alpha=0.4,
)

# ARS V2
plt.plot(episodes, v2_mean, color=V2_LINE, linewidth=2, label="ARS V2")
plt.fill_between(
    episodes,
    v2_mean - v2_std,
    v2_mean + v2_std,
    color=V2_SHADE,
    alpha=0.4,
)

plt.title("Swimmer v4 ARS V1 and V2", fontsize=14)
plt.xlabel("Episodes", fontsize=12)   # iterations equal episodes for Swimmer
plt.ylabel("AverageReward", fontsize=12)

plt.grid(alpha=0.25)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()
