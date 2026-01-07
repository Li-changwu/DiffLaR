#!/usr/bin/env python
"""
验证DiffLaR中Diffusion生成embedding的质量
计算生成的Steps Embedding与GT Steps Embedding的相似度
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.difflar import LitDiffLaR
from src.datasets.qsa import QSADataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的checkpoint路径")
    parser.add_argument("--num_samples", type=int, default=100, help="验证样本数")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"加载checkpoint: {args.ckpt_path}")
    
    # 加载checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=args.device, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    
    # 创建模型
    model = LitDiffLaR(
        model_kwargs=hparams["model_kwargs"],
        training_kwargs=hparams["training_kwargs"],
        all_config=hparams["all_config"],
    )
    
    # 加载保存的权重（只有diffusion部分）
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(args.device)
    
    print(f"模型加载完成，设备: {args.device}")
    print(f"num_inference_steps: {model.num_inference_steps}")
    print(f"max_latent_length: {model.max_latent_length}")
    
    # 加载数据
    data_module = QSADataModule(
        dataset_name="gsm",
        tiny_dataset=False,
        epoch_scaling=1,
        all_config=model.all_config,
    )
    data_module.all_config = model.all_config
    data_module.all_config.dataloader.batch_size = args.batch_size
    data_module.all_config.dataloader.num_workers = 0
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    # 收集统计
    all_cos_sim = []
    all_mse = []
    all_gen_norm = []
    all_gt_norm = []
    all_norm_ratio = []
    
    num_processed = 0
    
    print(f"\n开始验证Diffusion生成质量...")
    print("=" * 60)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if num_processed >= args.num_samples:
                break
                
            questions = batch["question"]
            steps = batch["steps"]
            
            # 计算相似度
            sim_dict = model.compute_embedding_similarity(questions, steps)
            
            all_cos_sim.append(sim_dict["cos_sim"])
            all_mse.append(sim_dict["mse"])
            all_gen_norm.append(sim_dict["gen_norm"])
            all_gt_norm.append(sim_dict["gt_norm"])
            all_norm_ratio.append(sim_dict["norm_ratio"])
            
            num_processed += len(questions)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Batch {batch_idx + 1}: cos_sim={sim_dict['cos_sim']:.4f}, "
                      f"mse={sim_dict['mse']:.4f}, norm_ratio={sim_dict['norm_ratio']:.4f}")
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    import numpy as np
    
    print(f"样本数: {num_processed}")
    print(f"\n余弦相似度 (越接近1越好):")
    print(f"  Mean: {np.mean(all_cos_sim):.4f}")
    print(f"  Std:  {np.std(all_cos_sim):.4f}")
    print(f"  Min:  {np.min(all_cos_sim):.4f}")
    print(f"  Max:  {np.max(all_cos_sim):.4f}")
    
    print(f"\nMSE (越小越好):")
    print(f"  Mean: {np.mean(all_mse):.4f}")
    print(f"  Std:  {np.std(all_mse):.4f}")
    
    print(f"\n范数比较:")
    print(f"  GT范数均值:   {np.mean(all_gt_norm):.4f}")
    print(f"  生成范数均值: {np.mean(all_gen_norm):.4f}")
    print(f"  范数比例 (生成/GT): {np.mean(all_norm_ratio):.4f}")
    
    # 诊断
    print("\n" + "=" * 60)
    print("诊断分析")
    print("=" * 60)
    
    mean_cos_sim = np.mean(all_cos_sim)
    if mean_cos_sim < 0.3:
        print("⚠️ 余弦相似度很低 (<0.3): Diffusion生成的方向与GT差异很大")
        print("   可能原因: 训练不充分 / 推理步数与训练不匹配 / 模型架构问题")
    elif mean_cos_sim < 0.7:
        print("⚠️ 余弦相似度中等 (0.3-0.7): Diffusion有一定生成能力但不准确")
    else:
        print("✓ 余弦相似度较高 (>0.7): Diffusion生成方向基本正确")
    
    mean_norm_ratio = np.mean(all_norm_ratio)
    if mean_norm_ratio < 0.5 or mean_norm_ratio > 2.0:
        print(f"⚠️ 范数比例异常 ({mean_norm_ratio:.2f}): 生成向量的尺度与GT不匹配")
    else:
        print(f"✓ 范数比例正常 ({mean_norm_ratio:.2f})")


if __name__ == "__main__":
    main()
