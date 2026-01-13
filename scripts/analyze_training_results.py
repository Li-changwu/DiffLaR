import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load training results
with open('/root/autodl-tmp/colar/logs/difflar_fused/qsa-gsm/20260112-141115_404344/train.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("DiffLaR Fused Training Analysis")
print("=" * 80)

# Basic statistics
n_samples = len(data)
# Find max number of epochs across all samples
n_epochs = max(len(sample_data['acc']) for sample_data in data.values()) if data else 0

print(f"\n📊 Dataset Overview:")
print(f"  - Total samples: {n_samples}")
print(f"  - Epochs trained: {n_epochs}")

# Analyze accuracy progression
epoch_accuracies = []
for epoch_idx in range(n_epochs):
    epoch_acc = []
    for sample_id, sample_data in data.items():
        if epoch_idx < len(sample_data['acc']):
            epoch_acc.append(sample_data['acc'][epoch_idx])
    epoch_accuracies.append(np.mean(epoch_acc) if epoch_acc else 0)

print(f"\n📈 Accuracy Progression Across Epochs:")
for i, acc in enumerate(epoch_accuracies):
    print(f"  Epoch {i+1:2d}: {acc*100:6.2f}%")

# Calculate improvement
if len(epoch_accuracies) > 1:
    initial_acc = epoch_accuracies[0]
    final_acc = epoch_accuracies[-1]
    improvement = final_acc - initial_acc
    print(f"\n💡 Training Improvement:")
    print(f"  - Initial accuracy (Epoch 1): {initial_acc*100:.2f}%")
    print(f"  - Final accuracy (Epoch {n_epochs}): {final_acc*100:.2f}%")
    print(f"  - Absolute improvement: {improvement*100:+.2f}%")
    print(f"  - Relative improvement: {(improvement/max(initial_acc, 0.01))*100:+.2f}%")

# Analyze output quality evolution
print(f"\n📝 Output Quality Analysis:")
print(f"\n  Sample predictions across epochs (first 3 samples):")
for sample_idx, (sample_id, sample_data) in enumerate(list(data.items())[:3]):
    print(f"\n  Sample {sample_idx+1} (ID: {sample_id}):")
    print(f"    Question: {sample_data['question'][:80]}...")
    print(f"    Ground truth answer: {sample_data['answer']}")
    print(f"\n    Predictions by epoch:")
    
    for epoch_idx in range(min(5, n_epochs)):  # First 5 epochs
        pred = sample_data['pred_answer'][epoch_idx]
        correct = "✓" if sample_data['acc'][epoch_idx] == 1.0 else "✗"
        print(f"      Epoch {epoch_idx+1:2d}: {pred[:60]:<60} {correct}")
    
    if n_epochs > 5:
        print(f"      ...")
        for epoch_idx in range(n_epochs-3, n_epochs):  # Last 3 epochs
            pred = sample_data['pred_answer'][epoch_idx]
            correct = "✓" if sample_data['acc'][epoch_idx] == 1.0 else "✗"
            print(f"      Epoch {epoch_idx+1:2d}: {pred[:60]:<60} {correct}")

# Analyze output length evolution
print(f"\n📏 Output Length Evolution:")
avg_lengths = []
for epoch_idx in range(n_epochs):
    epoch_lengths = [sample_data['output_length'][epoch_idx] for sample_data in data.values() if epoch_idx < len(sample_data['output_length'])]
    avg_lengths.append(np.mean(epoch_lengths) if epoch_lengths else 0)

print(f"  Average output length by epoch:")
for i, length in enumerate(avg_lengths):
    print(f"    Epoch {i+1:2d}: {length:.1f} tokens")

# Detect phase transition (if any)
phase_transition_epoch = None
for i in range(len(avg_lengths)-1):
    if avg_lengths[i] > 10 and avg_lengths[i+1] < 10:
        phase_transition_epoch = i+1
        break

if phase_transition_epoch:
    print(f"\n  ⚠️  Phase transition detected at epoch {phase_transition_epoch}")
    print(f"      Output length changed from {avg_lengths[phase_transition_epoch-1]:.1f} to {avg_lengths[phase_transition_epoch]:.1f} tokens")

# Analyze answer patterns
print(f"\n🎯 Answer Pattern Analysis:")
for epoch_idx in [0, n_epochs//2, n_epochs-1]:
    print(f"\n  Epoch {epoch_idx+1}:")
    
    # Count answer types
    coherent_count = 0
    gibberish_count = 0
    correct_count = 0
    
    for sample_data in data.values():
        if epoch_idx >= len(sample_data['pred_answer']):
            continue
        pred = sample_data['pred_answer'][epoch_idx]
        acc = sample_data['acc'][epoch_idx]
        
        if acc == 1.0:
            correct_count += 1
            coherent_count += 1
        elif pred.strip() and not any(c in pred for c in ['#', '=', 'Step', 'Answer:']):
            # Check if it's a reasonable number
            try:
                float(pred.strip())
                coherent_count += 1
            except:
                gibberish_count += 1
        else:
            if 'Answer:' in pred:
                coherent_count += 1
            else:
                gibberish_count += 1
    
    print(f"    Correct answers: {correct_count}/{n_samples} ({correct_count/n_samples*100:.1f}%)")
    print(f"    Coherent outputs: {coherent_count}/{n_samples} ({coherent_count/n_samples*100:.1f}%)")
    print(f"    Gibberish outputs: {gibberish_count}/{n_samples} ({gibberish_count/n_samples*100:.1f}%)")

# Overall assessment
print(f"\n" + "=" * 80)
print("🔍 Training Effectiveness Assessment:")
print("=" * 80)

if final_acc < 0.1:
    status = "❌ POOR - Model is not learning effectively"
elif final_acc < 0.3:
    status = "⚠️  WEAK - Model shows minimal learning"
elif final_acc < 0.5:
    status = "📊 MODERATE - Model is learning but needs improvement"
elif final_acc < 0.7:
    status = "✅ GOOD - Model is learning well"
else:
    status = "🌟 EXCELLENT - Model is learning very effectively"

print(f"\nStatus: {status}")

# Identify issues
issues = []
if final_acc < initial_acc:
    issues.append("Model accuracy is degrading over training")
if max(epoch_accuracies) - final_acc > 0.1:
    issues.append("Model may be overfitting or unstable")
if phase_transition_epoch:
    issues.append(f"Abrupt phase transition at epoch {phase_transition_epoch} suggests training instability")
if final_acc < 0.2:
    issues.append("Very low final accuracy suggests fundamental training problems")

if issues:
    print(f"\n⚠️  Issues Detected:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"\n✅ No major issues detected")

# Recommendations
print(f"\n💡 Recommendations:")
if final_acc < 0.2:
    print(f"  - Check loss convergence and gradient flow")
    print(f"  - Verify diffusion model is properly conditioned on input")
    print(f"  - Consider adjusting learning rate or diffusion timesteps")
elif improvement < 0:
    print(f"  - Model is degrading - check for training instabilities")
    print(f"  - Consider reducing learning rate or adding regularization")
elif improvement < 0.1:
    print(f"  - Limited improvement - may need more training or hyperparameter tuning")
    print(f"  - Consider increasing model capacity or adjusting diffusion schedule")
else:
    print(f"  - Continue training to see if accuracy continues improving")
    print(f"  - Monitor for overfitting on validation set")

print("\n" + "=" * 80)
