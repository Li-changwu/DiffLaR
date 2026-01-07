"""
生成DiffLaR论文图表的脚本
运行方式: python plot_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

def plot_steps_ablation():
    """Figure 3: 采样步数消融实验"""
    # 示例数据 - 需要替换为真实实验结果
    steps = [16, 32, 64, 128, 256, 512]
    accuracy = [28.4, 31.2, 33.8, 35.2, 35.6, 35.9]
    latency = [95, 185, 298, 512, 892, 1680]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = '#4285F4'  # Google Blue
    ax1.set_xlabel('Number of Denoising Steps (K)')
    ax1.set_ylabel('Accuracy (%)', color=color1)
    line1 = ax1.plot(steps, accuracy, 'o-', color=color1, linewidth=2, 
                     markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(steps)
    ax1.set_xticklabels(steps)
    ax1.set_ylim([25, 40])
    ax1.axhline(y=35.2, color=color1, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = '#EA4335'  # Google Red
    ax2.set_ylabel('Latency (ms)', color=color2)
    line2 = ax2.plot(steps, latency, 's--', color=color2, linewidth=2, 
                     markersize=8, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 2000])
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.set_title('Effect of Denoising Steps on GSM8K')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('steps_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('steps_ablation.png', dpi=300, bbox_inches='tight')
    print("Saved steps_ablation.pdf")

def plot_tsne(embeddings=None, labels=None):
    """Figure 4: t-SNE可视化"""
    # 示例数据 - 需要替换为真实嵌入
    np.random.seed(42)
    n_samples = 200
    
    if embeddings is None:
        # 模拟4类问题的嵌入分布
        centers = [(0, 0), (5, 5), (-5, 5), (0, -5)]
        embeddings_2d = []
        labels = []
        for i, (cx, cy) in enumerate(centers):
            x = np.random.randn(n_samples // 4) * 1.5 + cx
            y = np.random.randn(n_samples // 4) * 1.5 + cy
            embeddings_2d.extend(zip(x, y))
            labels.extend([i] * (n_samples // 4))
        embeddings_2d = np.array(embeddings_2d)
        labels = np.array(labels)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335']
    problem_types = ['Addition', 'Subtraction', 'Multiplication', 'Multi-step']
    
    for i, (color, ptype) in enumerate(zip(colors, problem_types)):
        mask = labels == i
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=color, label=ptype, alpha=0.7, s=50, edgecolors='white')
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Generated Reasoning Embeddings')
    ax.legend(loc='upper right', title='Problem Type')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsne.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('tsne.png', dpi=300, bbox_inches='tight')
    print("Saved tsne.pdf")

if __name__ == '__main__':
    plot_steps_ablation()
    plot_tsne()
    print("All figures generated!")
