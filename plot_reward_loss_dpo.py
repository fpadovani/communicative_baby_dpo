import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("./dpo_outputs_complete_synthetic/logs/training_metrics.csv")  # Replace with your actual CSV file path

# Plot 1: Loss over Steps
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["loss"], label="Loss", color="blue", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/loss_trend_dpo_synthetic.png")
plt.close()

# Plot 2: Rewards/Chosen and Rewards/Rejected over Steps
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["rewards/chosen"], label="Rewards/Chosen", color="green", linewidth=2)
plt.plot(df["step"], df["rewards/rejected"], label="Rewards/Rejected", color="red", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Rewards (Chosen vs Rejected) over Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./plots/rewards_trend_dpo_synthetic.png")
plt.close()
