import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_dir = '/root/autodl-tmp/colar/logs/difflar_fused/qsa-gsm/20260112-141115_404344'

# Load data
df = pd.read_csv(f'{log_dir}/loss_history.csv')

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('DiffLaR Fused Training Analysis - Loss vs Accuracy Paradox', fontsize=16, fontweight='bold')

# Plot 1: Total Loss over steps
ax1 = axes[0, 0]
ax1.plot(df['step'], df['total_loss'], alpha=0.3, linewidth=0.5, label='Total Loss')
# Moving average
window = 100
df['total_loss_ma'] = df['total_loss'].rolling(window=window, min_periods=1).mean()
ax1.plot(df['step'], df['total_loss_ma'], color='red', linewidth=2, label=f'{window}-step MA')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Total Loss')
ax1.set_title('Total Loss Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mark stage transition
stage_change = df[df['stage'] != df['stage'].shift()].index
for idx in stage_change[1:]:
    ax1.axvline(x=df.loc[idx, 'step'], color='green', linestyle='--', alpha=0.5, label='Stage Change' if idx == stage_change[1] else '')

# Plot 2: Loss components
ax2 = axes[0, 1]
df['diff_loss_ma'] = df['diffusion_loss'].rolling(window=window, min_periods=1).mean()
df['ans_loss_ma'] = df['answer_loss'].rolling(window=window, min_periods=1).mean()
ax2.plot(df['step'], df['diff_loss_ma'], label='Diffusion Loss', linewidth=2)
ax2.plot(df['step'], df['ans_loss_ma'], label='Answer Loss', linewidth=2)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Components (100-step MA)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss by epoch vs Accuracy
ax3 = axes[1, 0]
epoch_stats = df.groupby('epoch').agg({
    'total_loss': 'mean',
    'diffusion_loss': 'mean',
    'answer_loss': 'mean'
}).reset_index()

# Accuracy data from previous analysis
accuracy_data = [0.00, 0.00, 0.00, 0.00, 0.00, 2.67, 4.28, 3.21, 2.94, 4.01, 
                 3.21, 4.41, 3.34, 3.88, 4.55, 3.74, 4.68, 3.34, 3.48, 4.68]

ax3_twin = ax3.twinx()
ax3.plot(epoch_stats['epoch'], epoch_stats['total_loss'], 'b-o', label='Total Loss', linewidth=2, markersize=6)
ax3_twin.plot(range(len(accuracy_data)), accuracy_data, 'r-s', label='Accuracy (%)', linewidth=2, markersize=6)

ax3.set_xlabel('Epoch')
ax3.set_ylabel('Total Loss', color='b')
ax3_twin.set_ylabel('Accuracy (%)', color='r')
ax3.set_title('Loss Decreases BUT Accuracy Stays Near 0%')
ax3.tick_params(axis='y', labelcolor='b')
ax3_twin.tick_params(axis='y', labelcolor='r')
ax3.grid(True, alpha=0.3)

# Add legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Plot 4: Rollout ratio and loss variance
ax4 = axes[1, 1]
stage2_data = df[df['stage'] == 2].copy()
if len(stage2_data) > 0:
    # Group by rollout ratio
    rollout_groups = stage2_data.groupby('rollout_ratio').agg({
        'total_loss': ['mean', 'std', 'count']
    }).reset_index()
    rollout_groups.columns = ['rollout_ratio', 'loss_mean', 'loss_std', 'count']
    
    ax4.errorbar(rollout_groups['rollout_ratio'], rollout_groups['loss_mean'], 
                 yerr=rollout_groups['loss_std'], fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax4.set_xlabel('Rollout Ratio')
    ax4.set_ylabel('Total Loss (mean ± std)')
    ax4.set_title('Loss vs Rollout Ratio (Stage 2)')
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No Stage 2 data with rollout', ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/colar/logs/difflar_fused/qsa-gsm/20260107-223135_423644/training_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved to: logs/difflar_fused/qsa-gsm/20260107-223135_423644/training_analysis.png")

# Additional insight plot: Output length vs accuracy
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

# Output length data from previous analysis
output_lengths = [16.0] * 5 + [5.2, 4.1] + [4.1] * 13

ax_twin = ax.twinx()
epochs = range(len(accuracy_data))
ax.plot(epochs, output_lengths[:len(epochs)], 'g-^', label='Avg Output Length', linewidth=2, markersize=8)
ax_twin.plot(epochs, accuracy_data, 'r-s', label='Accuracy (%)', linewidth=2, markersize=6)

ax.set_xlabel('Epoch')
ax.set_ylabel('Output Length (tokens)', color='g')
ax_twin.set_ylabel('Accuracy (%)', color='r')
ax.set_title('Phase Transition at Epoch 5-6: Output Length Drops, Accuracy Appears Then Vanishes')
ax.tick_params(axis='y', labelcolor='g')
ax_twin.tick_params(axis='y', labelcolor='r')

# Mark the phase transition
ax.axvline(x=5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Phase Transition')
ax.axvspan(5, 6, alpha=0.2, color='orange')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{log_dir}/phase_transition.png', dpi=150, bbox_inches='tight')
print(f"✅ Phase transition plot saved to: {log_dir}/phase_transition.png")

plt.close('all')
print("\n📊 Visualization complete!")
