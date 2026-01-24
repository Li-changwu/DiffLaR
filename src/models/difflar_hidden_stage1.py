import random
from typing import Dict

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from ..modules.diffusion import LatentDiffusion


class LitDiffLaRHiddenStage1(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__()

        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.save_hyperparameters()

        difflar_config = model_kwargs.difflar_config

        self.hidden_size = difflar_config.get("hidden_size")
        if self.hidden_size is None:
            raise ValueError("Stage1 requires difflar_config.hidden_size (embedding/hidden dim).")

        self.latent_diffusion = LatentDiffusion(
            hidden_size=self.hidden_size,
            num_timesteps=difflar_config.get("num_timesteps", 1000),
            num_layers=difflar_config.get("denoiser_layers", 6),
            num_heads=difflar_config.get("denoiser_heads", 8),
            max_latent_length=difflar_config.get("max_latent_length", 256),
            schedule_type=difflar_config.get("noise_schedule", "linear"),
            dropout=difflar_config.get("dropout", 0.1),
            normalize_latent=difflar_config.get("normalize_latent", True),
            prediction_type=difflar_config.get("prediction_type", "flow"),
            sampler_type=difflar_config.get("sampler_type", "euler"),
            ddim_eta=difflar_config.get("ddim_eta", 0.0),
        )

        self.max_latent_length = difflar_config.get("max_latent_length", 256)
        self.max_condition_length = difflar_config.get("max_condition_length", 128)
        self.diffusion_loss_weight = difflar_config.get("diffusion_loss_weight", 2.0)
        self.alignment_loss_weight = difflar_config.get("alignment_loss_weight", 0.6)

        self.stage1_epochs = difflar_config.get("stage1_epochs", 10)

        self.stage1a_ratio = difflar_config.get("stage1a_ratio", 0.5)
        self.stage1a_train_inference_steps = difflar_config.get("stage1a_train_inference_steps", 10)
        self.stage1b_train_inference_steps = difflar_config.get("stage1b_train_inference_steps", 20)
        self.stage1b_high_noise_ratio = difflar_config.get("stage1b_high_noise_ratio", 0.5)

        self.stage1_self_cond_prob = difflar_config.get("stage1_self_cond_prob", 0.5)

    def configure_optimizers(self):
        kwargs = self.all_config.model.training_kwargs

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = kwargs.optimizer
        opt = __import__(optimizer.target.rsplit(".", 1)[0], fromlist=[optimizer.target.rsplit(".", 1)[1]])
        opt_cls = getattr(opt, optimizer.target.rsplit(".", 1)[1])
        return opt_cls(trainable_params, lr=optimizer.lr, weight_decay=optimizer.get("weight_decay", 0.0))

    def _pad_to_fixed_length(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        max_length: int,
    ):
        batch_size, cur_len, hidden = x.shape
        if cur_len > max_length:
            return x[:, :max_length, :], mask[:, :max_length]
        if cur_len < max_length:
            pad_len = max_length - cur_len
            pad_x = torch.zeros(batch_size, pad_len, hidden, device=x.device, dtype=x.dtype)
            pad_m = torch.zeros(batch_size, pad_len, device=mask.device, dtype=mask.dtype)
            return torch.cat([x, pad_x], dim=1), torch.cat([mask, pad_m], dim=1)
        return x, mask

    def _compute_alignment_loss(
        self,
        generated_hidden: torch.Tensor,
        gt_hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)
        mse_loss = F.mse_loss(
            generated_hidden * mask_expanded,
            gt_hidden * mask_expanded,
            reduction="sum",
        ) / (mask.sum() * generated_hidden.shape[-1] + 1e-8)

        gen_norm = F.normalize(generated_hidden, p=2, dim=-1)
        gt_norm = F.normalize(gt_hidden, p=2, dim=-1)
        cosine_sim = (gen_norm * gt_norm).sum(dim=-1)
        cosine_loss = ((1 - cosine_sim) * mask).sum() / (mask.sum() + 1e-8)
        return mse_loss + 0.1 * cosine_loss

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        q_embeds = batch["question_embeds"].float().to(self.device)
        gt_steps_hidden = batch["steps_hidden"].float().to(self.device)
        q_mask = batch["question_mask"].float().to(self.device)
        steps_mask = batch["steps_mask"].float().to(self.device)

        q_cond, q_mask_padded = self._pad_to_fixed_length(q_embeds, q_mask, self.max_condition_length)
        gt_steps_padded, steps_mask_padded = self._pad_to_fixed_length(
            gt_steps_hidden, steps_mask, self.max_latent_length
        )

        stage1a_epochs = int(self.stage1_epochs * self.stage1a_ratio)
        is_stage1b = self.current_epoch >= stage1a_epochs

        self.latent_diffusion.self_cond_prob = self.stage1_self_cond_prob

        t_range = None
        if is_stage1b and random.random() < self.stage1b_high_noise_ratio:
            t_range = (0.8, 1.0)

        diffusion_loss, _, _ = self.latent_diffusion(
            steps_embeds=gt_steps_padded,
            condition=q_cond,
            attention_mask=steps_mask_padded,
            condition_mask=q_mask_padded,
            use_self_cond=None,
            t_range=t_range,
        )

        train_inference_steps = self.stage1b_train_inference_steps if is_stage1b else self.stage1a_train_inference_steps
        generated_steps = self.latent_diffusion.generate(
            condition=q_cond,
            num_inference_steps=train_inference_steps,
            latent_length=self.max_latent_length,
            condition_mask=q_mask_padded,
            enable_grad=True,
            use_self_cond=True,
        )

        alignment_loss = self._compute_alignment_loss(generated_steps, gt_steps_padded, steps_mask_padded)

        total_loss = self.diffusion_loss_weight * diffusion_loss + self.alignment_loss_weight * alignment_loss

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "alignment_loss": alignment_loss,
        }

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log_dict(
            {"train/" + k: v for k, v in out.items()},
            sync_dist=True,
            prog_bar=True,
            batch_size=self.all_config.dataloader.batch_size,
        )
        return out["total_loss"]




