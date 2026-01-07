#!/usr/bin/env python
"""
快速验证归一化效果 - 无需完整训练
只需几个step就能观察到归一化是否解决了数值爆炸问题
"""
import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.difflar import LitDiffLaR
from src.datasets.qsa import QSADataModule
from omegaconf import OmegaConf


def create_minimal_config():
    """创建最小化配置用于快速测试"""
    config = OmegaConf.create({
        "args": {
            "workspace_path": "/root/autodl-tmp/colar",
            "model": "difflar",
            "dataset": "qsa",
            "no_log": True,
        },
        "dataloader": {
            "batch_size": 4,
            "val_batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "model": {
            "target": "src.models.difflar.LitDiffLaR",
            "model_kwargs": {
                "model_id": "Llama-3.2-1B-Instruct",
                "sft_method": "difflar",
                "chat_template": False,
                "do_lora": True,
                "lora_config": {"r": 32, "lora_alpha": 16},
                "difflar_config": {
                    "max_latent_length": 64,  # 缩小以加速
                    "num_timesteps": 100,     # 缩小以加速
                    "num_inference_steps": 20,
                    "train_inference_steps": 5,
                    "denoiser_layers": 2,     # 缩小以加速
                    "denoiser_heads": 4,
                    "dropout": 0.1,
                    "diffusion_loss_weight": 1.0,
                    "answer_loss_weight": 5.0,
                    "normalize_latent": True,  # 关键：启用归一化
                    "prediction_type": "epsilon",
                    "clamp_value": 3.0,
                },
                "answer_generation_config": {
                    "max_new_tokens": 8,
                    "do_sample": False,
                }
            },
            "training_kwargs": {
                "optimizer": {"target": "torch.optim.AdamW", "lr": 1e-4},
                "use_scheduler": False,
            }
        }
    })
    return config


def quick_verify(normalize_latent=True, num_steps=10, device="cuda:0"):
    """快速验证归一化效果"""
    print(f"\n{'='*60}")
    print(f"快速验证: normalize_latent={normalize_latent}")
    print(f"{'='*60}")
    
    # 创建配置
    config = create_minimal_config()
    config.model.model_kwargs.difflar_config.normalize_latent = normalize_latent
    
    # 创建模型
    print("创建模型...")
    model = LitDiffLaR(
        model_kwargs=config.model.model_kwargs,
        training_kwargs=config.model.training_kwargs,
        all_config=config,
    )
    model.to(device)
    model.train()
    
    # 创建数据
    print("加载数据...")
    data_module = QSADataModule(
        dataset_name="gsm",
        tiny_dataset=True,  # 使用小数据集加速
        all_config=config,
    )
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    
    # 手动估计归一化参数
    if normalize_latent:
        print("估计归一化参数...")
        all_embeds = []
        all_masks = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 10:
                    break
                steps = batch["steps"]
                steps_embeds, steps_mask = model.prepare_fixed_length_steps(steps)
                all_embeds.append(steps_embeds)
                all_masks.append(steps_mask)
        
        all_embeds = torch.cat(all_embeds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        mean, std, scale = model.latent_diffusion.estimate_latent_stats(all_embeds, all_masks)
        print(f"  Scale: {scale:.4f}")
        model._latent_stats_estimated = True
    
    # 创建优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # 训练几步并观察
    print(f"\n开始{num_steps}步训练观测...")
    print("-" * 60)
    
    train_iter = iter(train_loader)
    
    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        optimizer.zero_grad()
        
        # 只计算diffusion loss用于训练（避免answer loss的二次backward问题）
        questions = batch["question"]
        steps_text = batch["steps"]
        
        q_ids, q_mask = model.prepare_inputs(
            questions, padding_side="left", part="question",
            suffix=model.speed_template.format("auto") + model.thinking_separator,
        )
        query_emb = model.embedding(q_ids)
        gt_embeds, gt_mask = model.prepare_fixed_length_steps(steps_text)
        
        diff_loss, _, _ = model.latent_diffusion(
            steps_embeds=gt_embeds,
            condition=query_emb,
            attention_mask=gt_mask,
            condition_mask=q_mask.float(),
        )
        
        diff_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        # 每步都观测生成质量
        if step % 2 == 0:
            with torch.no_grad():
                questions = batch["question"][:2]
                steps = batch["steps"][:2]
                
                # 准备输入
                q_ids, q_mask = model.prepare_inputs(
                    questions, padding_side="left", part="question",
                    suffix=model.speed_template.format("auto") + model.thinking_separator,
                )
                query_emb = model.embedding(q_ids)
                gt_embeds, gt_mask = model.prepare_fixed_length_steps(steps)
                
                # 生成
                gen_embeds = model.latent_diffusion.generate(
                    condition=query_emb,
                    num_inference_steps=model.num_inference_steps,
                    latent_length=model.max_latent_length,
                    condition_mask=q_mask.float(),
                    clamp_value=model.clamp_value,
                )
                
                # 计算指标
                cos_sim = F.cosine_similarity(gen_embeds, gt_embeds, dim=-1)
                cos_sim_mean = (cos_sim * gt_mask).sum() / (gt_mask.sum() + 1e-8)
                gen_norm = gen_embeds.norm(dim=-1).mean()
                gt_norm = gt_embeds.norm(dim=-1).mean()
                
                print(f"Step {step:2d}: diff_loss={diff_loss.item():.4f}, "
                      f"cos_sim={cos_sim_mean:.4f}, "
                      f"gen_norm={gen_norm:.2f}, gt_norm={gt_norm:.2f}, "
                      f"ratio={gen_norm/gt_norm:.2f}")
    
    # 最终评估
    print("\n" + "-" * 60)
    print("最终生成质量评估:")
    
    with torch.no_grad():
        batch = next(iter(train_loader))
        questions = batch["question"]
        steps = batch["steps"]
        
        q_ids, q_mask = model.prepare_inputs(
            questions, padding_side="left", part="question",
            suffix=model.speed_template.format("auto") + model.thinking_separator,
        )
        query_emb = model.embedding(q_ids)
        gt_embeds, gt_mask = model.prepare_fixed_length_steps(steps)
        
        gen_embeds = model.latent_diffusion.generate(
            condition=query_emb,
            num_inference_steps=model.num_inference_steps,
            latent_length=model.max_latent_length,
            condition_mask=q_mask.float(),
            clamp_value=model.clamp_value,
        )
        
        cos_sim = F.cosine_similarity(gen_embeds, gt_embeds, dim=-1)
        cos_sim_mean = (cos_sim * gt_mask).sum() / (gt_mask.sum() + 1e-8)
        gen_norm = gen_embeds.norm(dim=-1).mean()
        gt_norm = gt_embeds.norm(dim=-1).mean()
        
        print(f"  余弦相似度: {cos_sim_mean:.4f}")
        print(f"  生成范数:   {gen_norm:.2f}")
        print(f"  GT范数:     {gt_norm:.2f}")
        print(f"  范数比例:   {gen_norm/gt_norm:.2f}")
        
        if gen_norm / gt_norm > 10:
            print("  ⚠️ 警告: 范数比例过大，可能存在数值爆炸！")
        elif gen_norm / gt_norm > 2:
            print("  ⚠️ 注意: 范数比例偏大")
        else:
            print("  ✓ 范数比例正常")
        
        if cos_sim_mean < 0.1:
            print("  ⚠️ 警告: 余弦相似度过低，生成质量差")
        elif cos_sim_mean < 0.5:
            print("  ⚠️ 注意: 余弦相似度偏低")
        else:
            print("  ✓ 余弦相似度良好")
    
    return {
        "cos_sim": cos_sim_mean.item(),
        "gen_norm": gen_norm.item(),
        "gt_norm": gt_norm.item(),
        "norm_ratio": (gen_norm / gt_norm).item(),
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "=" * 70)
    print("DiffLaR 归一化效果快速验证")
    print("=" * 70)
    
    # 对比测试
    print("\n>>> 测试1: 启用归一化 (normalize_latent=True)")
    result_with_norm = quick_verify(normalize_latent=True, num_steps=10, device=device)
    
    print("\n>>> 测试2: 禁用归一化 (normalize_latent=False)")
    result_without_norm = quick_verify(normalize_latent=False, num_steps=10, device=device)
    
    # 对比总结
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(f"{'指标':<20} {'启用归一化':<15} {'禁用归一化':<15}")
    print("-" * 50)
    print(f"{'余弦相似度':<20} {result_with_norm['cos_sim']:<15.4f} {result_without_norm['cos_sim']:<15.4f}")
    print(f"{'范数比例':<20} {result_with_norm['norm_ratio']:<15.2f} {result_without_norm['norm_ratio']:<15.2f}")
    
    if result_with_norm['norm_ratio'] < result_without_norm['norm_ratio'] * 0.5:
        print("\n✓ 归一化显著改善了数值稳定性！")
    else:
        print("\n⚠️ 归一化效果不明显，可能需要调整参数")


if __name__ == "__main__":
    main()
