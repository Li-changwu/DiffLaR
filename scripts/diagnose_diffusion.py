#!/usr/bin/env python
"""
诊断Diffusion生成过程中的数值问题
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


def diagnose_scheduler(scheduler):
    """检查scheduler的数值范围"""
    print("\n=== Scheduler诊断 ===")
    print(f"num_timesteps: {scheduler.num_timesteps}")
    print(f"betas范围: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"alphas范围: [{scheduler.alphas.min():.6f}, {scheduler.alphas.max():.6f}]")
    print(f"alphas_cumprod范围: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")
    print(f"sqrt_alphas_cumprod范围: [{scheduler.sqrt_alphas_cumprod.min():.6f}, {scheduler.sqrt_alphas_cumprod.max():.6f}]")
    print(f"sqrt_one_minus_alphas_cumprod范围: [{scheduler.sqrt_one_minus_alphas_cumprod.min():.6f}, {scheduler.sqrt_one_minus_alphas_cumprod.max():.6f}]")
    print(f"sqrt_recip_alphas范围: [{scheduler.sqrt_recip_alphas.min():.6f}, {scheduler.sqrt_recip_alphas.max():.6f}]")
    
    # 检查推理时间步
    timesteps = scheduler.get_timesteps(128)
    print(f"\n推理timesteps (128步): {timesteps[:5].tolist()} ... {timesteps[-5:].tolist()}")
    print(f"timesteps范围: [{timesteps.min()}, {timesteps.max()}]")


def diagnose_generation_process(model, questions, steps, device):
    """诊断生成过程中的数值变化"""
    print("\n=== 生成过程诊断 ===")
    
    # 准备输入
    question_input_ids, question_attention_mask = model.prepare_inputs(
        questions,
        padding_side="left",
        part="question",
        suffix=model.speed_template.format("auto") + model.thinking_separator,
    )
    query_embedding = model.embedding(question_input_ids)
    query_mask = question_attention_mask.float()
    
    # GT steps embedding
    gt_steps_embeds, gt_steps_mask = model.prepare_fixed_length_steps(steps)
    print(f"GT steps embedding范数: {gt_steps_embeds.norm(dim=-1).mean():.4f}")
    print(f"GT steps embedding每维标准差: {gt_steps_embeds.std():.6f}")
    
    # 开始生成过程
    batch_size = query_embedding.shape[0]
    scheduler = model.latent_diffusion.scheduler
    denoiser = model.latent_diffusion.denoiser
    latent_length = model.max_latent_length
    hidden_size = model.latent_diffusion.hidden_size
    
    # 初始噪声
    x_t = torch.randn(batch_size, latent_length, hidden_size, device=device)
    print(f"\n初始噪声 x_T 范数: {x_t.norm(dim=-1).mean():.4f}")
    
    timesteps = scheduler.get_timesteps(model.num_inference_steps).to(device)
    
    # 跟踪几个关键步骤
    check_points = [0, 1, 2, 10, 50, 100, 127]  # 索引
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = denoiser(x_t, t_batch, query_embedding, None, query_mask)
            
            if i in check_points:
                print(f"\n--- Step {i}, t={t} ---")
                print(f"  x_t范数: {x_t.norm(dim=-1).mean():.4f}")
                print(f"  predicted_noise范数: {predicted_noise.norm(dim=-1).mean():.4f}")
                
                # 检查denoise_step中的系数
                betas_t = scheduler.betas[t_batch].mean()
                sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[t_batch].mean()
                sqrt_recip_alphas_t = scheduler.sqrt_recip_alphas[t_batch].mean()
                
                print(f"  betas[t]: {betas_t:.6f}")
                print(f"  sqrt_1-alpha_bar[t]: {sqrt_one_minus_alpha_cumprod_t:.6f}")
                print(f"  sqrt_recip_alpha[t]: {sqrt_recip_alphas_t:.6f}")
            
            # 执行降噪
            add_noise = (i < len(timesteps) - 1)
            x_t = scheduler.denoise_step(x_t, predicted_noise, t_batch, add_noise)
            
            if i in check_points:
                print(f"  降噪后x_t范数: {x_t.norm(dim=-1).mean():.4f}")
    
    print(f"\n最终生成embedding范数: {x_t.norm(dim=-1).mean():.4f}")
    print(f"GT embedding范数: {gt_steps_embeds.norm(dim=-1).mean():.4f}")
    print(f"范数比例: {x_t.norm(dim=-1).mean() / gt_steps_embeds.norm(dim=-1).mean():.2f}")
    
    # 余弦相似度
    cos_sim = F.cosine_similarity(x_t, gt_steps_embeds, dim=-1)
    cos_sim_masked = (cos_sim * gt_steps_mask).sum() / (gt_steps_mask.sum() + 1e-8)
    print(f"余弦相似度: {cos_sim_masked:.4f}")


def diagnose_denoiser_on_training_data(model, questions, steps, device):
    """检查denoiser在训练分布上的表现"""
    print("\n=== Denoiser在训练分布上的诊断 ===")
    
    # 准备输入
    question_input_ids, question_attention_mask = model.prepare_inputs(
        questions, padding_side="left", part="question",
        suffix=model.speed_template.format("auto") + model.thinking_separator,
    )
    query_embedding = model.embedding(question_input_ids)
    query_mask = question_attention_mask.float()
    gt_steps_embeds, gt_steps_mask = model.prepare_fixed_length_steps(steps)
    
    scheduler = model.latent_diffusion.scheduler
    denoiser = model.latent_diffusion.denoiser
    batch_size = query_embedding.shape[0]
    
    # 测试不同时间步
    test_timesteps = [0, 100, 500, 800, 999]
    
    with torch.no_grad():
        for t_val in test_timesteps:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            
            # 加噪
            x_t, noise = scheduler.add_noise(gt_steps_embeds, t)
            
            # 预测噪声
            predicted_noise = denoiser(x_t, t, query_embedding, gt_steps_mask, query_mask)
            
            # 计算误差
            mse = F.mse_loss(predicted_noise, noise)
            cos_sim = F.cosine_similarity(predicted_noise.flatten(), noise.flatten(), dim=0)
            
            print(f"t={t_val}: MSE={mse:.4f}, CosSim={cos_sim:.4f}, "
                  f"pred_norm={predicted_noise.norm(dim=-1).mean():.4f}, "
                  f"noise_norm={noise.norm(dim=-1).mean():.4f}")


def main():
    device = "cuda:0"
    ckpt_path = "logs/difflar/qsa-gsm/20260105-225409_881501/checkpoints/last.ckpt"
    
    print(f"加载checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    
    model = LitDiffLaR(
        model_kwargs=hparams["model_kwargs"],
        training_kwargs=hparams["training_kwargs"],
        all_config=hparams["all_config"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)
    
    # 加载测试数据
    data_module = QSADataModule(
        dataset_name="gsm", tiny_dataset=False, epoch_scaling=1,
        all_config=model.all_config,
    )
    data_module.all_config.dataloader.batch_size = 2
    data_module.all_config.dataloader.num_workers = 0
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    batch = next(iter(test_loader))
    questions = batch["question"]
    steps = batch["steps"]
    
    # 诊断
    diagnose_scheduler(model.latent_diffusion.scheduler)
    diagnose_denoiser_on_training_data(model, questions, steps, device)
    diagnose_generation_process(model, questions, steps, device)


if __name__ == "__main__":
    main()
