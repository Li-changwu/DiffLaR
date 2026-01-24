from typing import Dict, List

import torch

from .model_base import LitCoTModelBase
from ..modules.diffusion import LatentDiffusion
from .components.frozen_llm_answer_head import FrozenLLMAnswerHead


class LitDiffLaRHiddenStage2(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )

        if bool(model_kwargs.get("do_lora", False)):
            raise ValueError("Stage2 forbids LoRA. Set model.model_kwargs.do_lora=false")

        difflar_config = model_kwargs.difflar_config

        self.hidden_size = self.llm.config.hidden_size

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
        self.num_inference_steps = difflar_config.get("num_inference_steps", 10)

        self._freeze_llm_all()

        self.answer_head = FrozenLLMAnswerHead(self.llm, self.tokenizer, self.embedding)

        self.stage1_diffusion_ckpt_path = model_kwargs.get("stage1_diffusion_ckpt_path")
        if self.stage1_diffusion_ckpt_path:
            self._load_stage1_diffusion_weights(self.stage1_diffusion_ckpt_path)

    def _freeze_llm_all(self):
        for p in self.llm.parameters():
            p.requires_grad = False
        if self.embedding is not None:
            for p in self.embedding.parameters():
                p.requires_grad = False

    def _load_stage1_diffusion_weights(self, ckpt_path: str):
        ckpt_path = str(ckpt_path).strip()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        diffusion_sd = {}
        for k, v in state_dict.items():
            if k.startswith("latent_diffusion."):
                diffusion_sd[k[len("latent_diffusion.") :]] = v
            elif k.startswith("model.latent_diffusion."):
                diffusion_sd[k[len("model.latent_diffusion.") :]] = v

        if not diffusion_sd:
            raise ValueError(
                f"No latent_diffusion weights found in {ckpt_path}. Expected keys like 'latent_diffusion.*'."
            )

        self.latent_diffusion.load_state_dict(diffusion_sd, strict=False)

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

    def _compute_query_embeds(self, questions: List[str]):
        q_input_ids, q_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        q_embeds = self.embedding(q_input_ids)
        q_embeds = q_embeds * q_attention_mask.unsqueeze(-1)
        return q_embeds, q_attention_mask

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        questions = batch["question"]
        answers = batch["answer"]

        q_embeds, q_attn = self._compute_query_embeds(questions)

        q_mask_f = q_attn.float()
        q_cond, q_cond_mask = self._pad_to_fixed_length(q_embeds, q_mask_f, self.max_condition_length)

        z_steps = self.latent_diffusion.generate(
            condition=q_cond,
            num_inference_steps=self.num_inference_steps,
            latent_length=self.max_latent_length,
            condition_mask=q_cond_mask,
            enable_grad=True,
            use_self_cond=True,
            debug_nan=True,
        )

        if torch.isnan(z_steps).any() or torch.isinf(z_steps).any():
            q_len = q_attn.sum(dim=1)
            q_cond_mask_sum = q_cond_mask.sum(dim=1) if q_cond_mask is not None else None
            stats = {
                "q_embeds": (q_embeds.min().item(), q_embeds.max().item(), q_embeds.mean().item()),
                "q_cond": (q_cond.min().item(), q_cond.max().item(), q_cond.mean().item()),
                "z_steps": (z_steps.min().item(), z_steps.max().item(), z_steps.mean().item()),
            }
            print(
                "[FATAL] NaN/Inf detected in z_steps. "
                f"batch_idx={getattr(self, '_last_batch_idx', 'NA')} "
                f"q_len(min/max)={q_len.min().item()}/{q_len.max().item()} "
                f"q_cond_mask_sum(min/max)={(q_cond_mask_sum.min().item() if q_cond_mask_sum is not None else 'NA')}/"
                f"{(q_cond_mask_sum.max().item() if q_cond_mask_sum is not None else 'NA')} "
                f"stats={stats} "
                f"sample_q={questions[0][:200]} sample_a={answers[0][:200]}"
            )
            raise RuntimeError("NaN/Inf detected in z_steps (latent diffusion generate). Stopping for debug.")

        answer_loss = self.answer_head.compute_loss(
            q_embeds=q_embeds,
            q_attention_mask=q_attn,
            z_steps=z_steps,
            answers=answers,
        )

        return {
            "total_loss": answer_loss,
            "answer_loss": answer_loss,
        }

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        self._last_batch_idx = batch_idx
        out = self.forward(batch)
        self.log_dict(
            {"train/" + k: v for k, v in out.items()},
            sync_dist=True,
            prog_bar=True,
            batch_size=self.all_config.dataloader.batch_size,
        )
        return out["total_loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.forward(batch)
        log_dict = {"val/" + k: v for k, v in out.items()}
        log_dict["monitor"] = (-out["total_loss"]).detach()
        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=len(batch["question"]),
        )
        return log_dict
