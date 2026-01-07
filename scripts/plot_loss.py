import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/root/autodl-tmp/colar/logs/difflar/qsa-gsm/20260105-225409_881501/loss_history.csv"
df = pd.read_csv(csv_path)

plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['total_loss'], linewidth=0.8)
plt.xlabel('Steps')
plt.ylabel('Total Loss')
plt.title('Total Loss vs Steps')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/root/autodl-tmp/colar/logs/difflar/qsa-gsm/20260105-225409_881501/total_loss.png", dpi=150)
plt.show()
print("图表已保存到 total_loss.png")
