import csv
import os
import matplotlib.pyplot as plt


# =========================
# Create plot folder
# =========================

os.makedirs("plots", exist_ok=True)


# =========================
# Read CSV
# =========================

episodes = []
win_rates = []
losses = []
rewards = []
q_values = []
lengths = []


with open("training_results.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        episodes.append(
            int(row["episode"])
        )

        win_rates.append(
            float(row["win_rate"])
        )

        losses.append(
            float(row["avg_loss"])
        )

        rewards.append(
            float(row["avg_reward"])
        )

        q_values.append(
            float(row["avg_q"])
        )

        lengths.append(
            float(row["avg_episode_length"])
        )


# =========================
# 1. Win Rate
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    win_rates
)

plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.title("Training Win Rate")

plt.grid()

plt.savefig(
    "plots/training_win_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 2. Training Loss
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    losses
)

plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("Training Loss")

plt.grid()

plt.savefig(
    "plots/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 3. Episode Reward
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    rewards
)

plt.xlabel("Episode")
plt.ylabel("Average Episode Reward")
plt.title("Episode Reward")

plt.grid()

plt.savefig(
    "plots/episode_reward.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 4. Q Value
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    q_values
)

plt.xlabel("Episode")
plt.ylabel("Average Max Q")
plt.title("Average Max Q Value")

plt.grid()

plt.savefig(
    "plots/q_value.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 5. Episode Length
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    lengths
)

plt.xlabel("Episode")
plt.ylabel("Average Episode Length")
plt.title("Episode Length")

plt.grid()

plt.savefig(
    "plots/episode_length.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("All plots saved successfully.")