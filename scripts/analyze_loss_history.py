import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load loss history
df = pd.read_csv('/root/autodl-tmp/colar/logs/difflar_fused/qsa-gsm/20260112-113048_530615/loss_history.csv')

print("=" * 80)
print("Loss History Analysis - DiffLaR Fused")
print("=" * 80)

print(f"\n📊 Dataset Overview:")
print(f"  - Total training steps: {len(df)}")
print(f"  - Epochs: {df['epoch'].max() + 1}")
print(f"  - Stages: {df['stage'].unique()}")
print(f"  - Steps per epoch: {len(df) // (df['epoch'].max() + 1)}")

print(f"\n📈 Loss Statistics:")
print(f"\n  Total Loss:")
print(f"    Initial: {df['total_loss'].iloc[0]:.4f}")
print(f"    Final: {df['total_loss'].iloc[-1]:.4f}")
print(f"    Min: {df['total_loss'].min():.4f}")
print(f"    Max: {df['total_loss'].max():.4f}")
print(f"    Mean: {df['total_loss'].mean():.4f}")
print(f"    Std: {df['total_loss'].std():.4f}")

print(f"\n  Diffusion Loss:")
print(f"    Initial: {df['diffusion_loss'].iloc[0]:.4f}")
print(f"    Final: {df['diffusion_loss'].iloc[-1]:.4f}")
print(f"    Min: {df['diffusion_loss'].min():.4f}")
print(f"    Max: {df['diffusion_loss'].max():.4f}")
print(f"    Mean: {df['diffusion_loss'].mean():.4f}")

print(f"\n  Answer Loss:")
print(f"    Initial: {df['answer_loss'].iloc[0]:.4f}")
print(f"    Final: {df['answer_loss'].iloc[-1]:.4f}")
print(f"    Min: {df['answer_loss'].min():.4f}")
print(f"    Max: {df['answer_loss'].max():.4f}")
print(f"    Mean: {df['answer_loss'].mean():.4f}")

# Analyze by epoch
print(f"\n📊 Loss by Epoch:")
epoch_stats = df.groupby('epoch').agg({
    'total_loss': ['mean', 'min', 'max', 'std'],
    'diffusion_loss': 'mean',
    'answer_loss': 'mean'
})

print(f"\n{'Epoch':<6} {'Total Loss':<12} {'Diff Loss':<12} {'Ans Loss':<12} {'Std':<10}")
print("-" * 60)
for epoch in sorted(df['epoch'].unique()):
    epoch_data = df[df['epoch'] == epoch]
    total_mean = epoch_data['total_loss'].mean()
    diff_mean = epoch_data['diffusion_loss'].mean()
    ans_mean = epoch_data['answer_loss'].mean()
    total_std = epoch_data['total_loss'].std()
    print(f"{epoch:<6} {total_mean:<12.4f} {diff_mean:<12.4f} {ans_mean:<12.4f} {total_std:<10.4f}")

# Detect anomalies
print(f"\n🔍 Anomaly Detection:")

# Check for loss spikes
mean_loss = df['total_loss'].mean()
std_loss = df['total_loss'].std()
spikes = df[df['total_loss'] > mean_loss + 3 * std_loss]
if len(spikes) > 0:
    print(f"  ⚠️  {len(spikes)} loss spikes detected (>3σ):")
    print(f"      Steps: {spikes['step'].tolist()[:10]}...")
    print(f"      Max spike: {spikes['total_loss'].max():.4f} at step {spikes['total_loss'].idxmax()}")
else:
    print(f"  ✅ No major loss spikes detected")

# Check for NaN or Inf
nan_count = df['total_loss'].isna().sum()
inf_count = np.isinf(df['total_loss']).sum()
if nan_count > 0 or inf_count > 0:
    print(f"  ⚠️  NaN values: {nan_count}, Inf values: {inf_count}")
else:
    print(f"  ✅ No NaN or Inf values")

# Convergence analysis
print(f"\n📉 Convergence Analysis:")
first_1000 = df.head(1000)['total_loss'].mean()
last_1000 = df.tail(1000)['total_loss'].mean()
improvement = first_1000 - last_1000
improvement_pct = (improvement / first_1000) * 100

print(f"  First 1000 steps avg: {first_1000:.4f}")
print(f"  Last 1000 steps avg: {last_1000:.4f}")
print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

if improvement > 0:
    print(f"  ✅ Loss is decreasing - model is learning")
else:
    print(f"  ⚠️  Loss is not decreasing - training may have issues")

# Check stage-wise performance
print(f"\n🎯 Stage Analysis:")
for stage in sorted(df['stage'].unique()):
    stage_data = df[df['stage'] == stage]
    print(f"\n  Stage {stage}:")
    print(f"    Steps: {len(stage_data)}")
    print(f"    Total Loss: {stage_data['total_loss'].mean():.4f} ± {stage_data['total_loss'].std():.4f}")
    print(f"    Diffusion Loss: {stage_data['diffusion_loss'].mean():.4f}")
    print(f"    Answer Loss: {stage_data['answer_loss'].mean():.4f}")
    
    # Check if rollout_ratio changes in this stage
    if 'rollout_ratio' in stage_data.columns:
        rollout_values = stage_data['rollout_ratio'].unique()
        if len(rollout_values) > 1:
            print(f"    Rollout ratios: {sorted(rollout_values)}")

# Loss component analysis
print(f"\n🧩 Loss Component Analysis:")
total_avg = df['total_loss'].mean()
diff_avg = df['diffusion_loss'].mean()
ans_avg = df['answer_loss'].mean()

# Note: total_loss might be a weighted sum, not direct sum
print(f"  Average Total Loss: {total_avg:.4f}")
print(f"  Average Diffusion Loss: {diff_avg:.4f} ({diff_avg/total_avg*100:.1f}% of total)")
print(f"  Average Answer Loss: {ans_avg:.4f} ({ans_avg/total_avg*100:.1f}% of total)")

# Check correlation between losses
corr_diff_ans = df['diffusion_loss'].corr(df['answer_loss'])
print(f"\n  Correlation (Diffusion ↔ Answer): {corr_diff_ans:.3f}")

# Final assessment
print(f"\n" + "=" * 80)
print("🔍 Training Diagnosis:")
print("=" * 80)

issues = []
insights = []

# Check overall convergence
if improvement < 0.5:
    issues.append("Minimal loss reduction - training is not converging well")
else:
    insights.append(f"Loss reduced by {improvement:.2f} ({improvement_pct:.1f}%) - shows learning")

# Check stability
if df['total_loss'].std() > df['total_loss'].mean() * 0.5:
    issues.append("High loss variance - training is unstable")

# Check if answer loss is improving
first_ans = df.head(1000)['answer_loss'].mean()
last_ans = df.tail(1000)['answer_loss'].mean()
if last_ans > first_ans:
    issues.append(f"Answer loss increased from {first_ans:.2f} to {last_ans:.2f}")
else:
    insights.append(f"Answer loss improved from {first_ans:.2f} to {last_ans:.2f}")

# Check diffusion loss
first_diff = df.head(1000)['diffusion_loss'].mean()
last_diff = df.tail(1000)['diffusion_loss'].mean()
if last_diff > first_diff:
    issues.append(f"Diffusion loss increased from {first_diff:.2f} to {last_diff:.2f}")
else:
    insights.append(f"Diffusion loss improved from {first_diff:.2f} to {last_diff:.2f}")

if issues:
    print(f"\n⚠️  Issues Identified:")
    for issue in issues:
        print(f"  - {issue}")

if insights:
    print(f"\n✅ Positive Observations:")
    for insight in insights:
        print(f"  - {insight}")

# Compare with accuracy results
print(f"\n💡 Correlation with Accuracy Results:")
print(f"  - Loss shows learning (reduced by {improvement_pct:.1f}%)")
print(f"  - BUT accuracy remains near 0% throughout training")
print(f"  - This suggests: Loss is optimizing, but not the right objective")
print(f"  - Possible causes:")
print(f"    1. Diffusion loss dominates, but doesn't enforce correct reasoning")
print(f"    2. Answer loss is too weak or improperly weighted")
print(f"    3. Model learns to generate plausible text, not correct answers")
print(f"    4. Latent space doesn't capture reasoning structure")

print("\n" + "=" * 80)
