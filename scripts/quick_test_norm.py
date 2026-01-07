#!/usr/bin/env python
"""简化版归一化验证脚本"""
import sys
sys.path.insert(0, '/root/autodl-tmp/colar')

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.difflar import LitDiffLaR
from src.datasets.qsa import QSADataModule

def test(normalize_latent, device="cuda:0"):
    print(f"\n{'='*50}")
    print(f"测试: normalize_latent={normalize_latent}")
    print(f"{'='*50}")
    
    config = OmegaConf.create({
        "args": {"workspace_path": "/root/autodl-tmp/colar", "no_log": True},
        "dataloader": {"batch_size": 4, "num_workers": 0, "persistent_workers": False, "pin_memory": False},
        "model": {
            "model_kwargs": {
                "model_id": "Llama-3.2-1B-Instruct", "sft_method": "difflar",
                "chat_template": False, "do_lora": True,
                "lora_config": {"r": 32, "lora_alpha": 16},
                "difflar_config": {
                    "max_latent_length": 64, "num_timesteps": 100,
                    "num_inference_steps": 20, "train_inference_steps": 5,
                    "denoiser_layers": 2, "denoiser_heads": 4, "dropout": 0.1,
                    "normalize_latent": normalize_latent, "clamp_value": 3.0,
                },
                "answer_generation_config": {"max_new_tokens": 8, "do_sample": False}
            },
            "training_kwargs": {"optimizer": {"target": "torch.optim.AdamW", "lr": 1e-4}}
        }
    })
    
    model = LitDiffLaR(config.model.model_kwargs, config.model.training_kwargs, config)
    model.to(device).eval()
    
    data_module = QSADataModule("gsm", tiny_dataset=True, all_config=config)
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    
    # 估计归一化参数
    if normalize_latent:
        with torch.no_grad():
            embeds, masks = [], []
            for i, b in enumerate(loader):
                if i >= 5: break
                e, m = model.prepare_fixed_length_steps(b["steps"])
                embeds.append(e); masks.append(m)
            model.latent_diffusion.estimate_latent_stats(torch.cat(embeds), torch.cat(masks))
    
    # 测试生成
    batch = next(iter(loader))
    with torch.no_grad():
        q_ids, q_mask = model.prepare_inputs(batch["question"], "left", "question",
            suffix=model.speed_template.format("auto") + model.thinking_separator)
        query_emb = model.embedding(q_ids)
        gt_embeds, gt_mask = model.prepare_fixed_length_steps(batch["steps"])
        
        gen_embeds = model.latent_diffusion.generate(
            condition=query_emb, num_inference_steps=20,
            latent_length=64, condition_mask=q_mask.float(), clamp_value=3.0)
        
        cos_sim = F.cosine_similarity(gen_embeds, gt_embeds, dim=-1)
        cos_sim_mean = (cos_sim * gt_mask).sum() / (gt_mask.sum() + 1e-8)
        gen_norm = gen_embeds.norm(dim=-1).mean()
        gt_norm = gt_embeds.norm(dim=-1).mean()
        
        print(f"生成范数: {gen_norm:.2f}, GT范数: {gt_norm:.2f}")
        print(f"范数比例: {gen_norm/gt_norm:.2f}")
        print(f"余弦相似度: {cos_sim_mean:.4f}")
        return gen_norm/gt_norm

if __name__ == "__main__":
    r1 = test(True)
    r2 = test(False)
    print(f"\n{'='*50}")
    print(f"对比: 启用归一化ratio={r1:.2f}, 禁用ratio={r2:.2f}")
    print(f"改善倍数: {r2/r1:.1f}x" if r1 < r2 else "归一化效果不明显")
