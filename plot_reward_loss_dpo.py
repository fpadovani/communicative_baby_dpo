import pandas as pd
import matplotlib.pyplot as plt

# Load both training logs
df_synth = pd.read_csv("./dpo_outputs_complete_synthetic/logs/training_metrics.csv")
df_real = pd.read_csv("./dpo_outputs_complete/logs/training_metrics.csv")

# Set font sizes and line width
label_fontsize = 16
tick_fontsize = 14
line_width = 1.2
legend_fontsize = 18

# Plot 1: Loss over Steps
plt.figure(figsize=(10, 6))
plt.plot(df_synth["step"], df_synth["loss"], label="Synthetic DPO", color="blue", linewidth=line_width)
plt.plot(df_real["step"], df_real["loss"], label="Naturalistic DPO", color="orange", linewidth=line_width)
plt.xlabel("Step", fontsize=label_fontsize, labelpad=10)
plt.ylabel("Loss", fontsize=label_fontsize, labelpad=10)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize = legend_fontsize)
plt.tight_layout()
plt.savefig("./plots/loss_trend_dpo_comparison.png")
plt.close()

# Plot 2: Rewards over Steps
plt.figure(figsize=(10, 6))
plt.plot(df_synth["step"], df_synth["rewards/chosen"], label="Synthetic - Rewards/Chosen", color="#90ee90", linewidth=line_width)
plt.plot(df_synth["step"], df_synth["rewards/rejected"], label="Synthetic - Rewards/Rejected", color="orange", linewidth=line_width)
plt.plot(df_real["step"], df_real["rewards/chosen"], label="Naturalistic - Rewards/Chosen", color="darkgreen", linewidth=line_width)
plt.plot(df_real["step"], df_real["rewards/rejected"], label="Naturalistic - Rewards/Rejected", color="darkred", linewidth=line_width)
plt.xlabel("Step", fontsize=label_fontsize, labelpad=10)
plt.ylabel("Reward", fontsize=label_fontsize, labelpad=10)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize = 15, loc='lower left')
plt.tight_layout()
plt.savefig("./plots/rewards_trend_dpo_comparison.png")
plt.close()



'''# Plot 1: Loss over Steps
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
plt.close()'''
