from typing import List

import torch
import torch.nn as nn

from ...utils.utils import get_position_ids_from_attention_mask


class FrozenLLMAnswerHead(nn.Module):
    def __init__(self, llm, tokenizer, embedding):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.embedding = embedding

    def compute_loss(
        self,
        q_embeds: torch.Tensor,
        q_attention_mask: torch.Tensor,
        z_steps: torch.Tensor,
        answers: List[str],
    ) -> torch.Tensor:
        batch_size = q_embeds.shape[0]

        answer_inputs = self.tokenizer(
            answers,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
        )
        if self.tokenizer.eos_token_id is not None:
            eos = torch.full(
                (answer_inputs["input_ids"].shape[0], 1),
                self.tokenizer.eos_token_id,
                dtype=answer_inputs["input_ids"].dtype,
            )
            eos_mask = torch.ones(
                (answer_inputs["attention_mask"].shape[0], 1),
                dtype=answer_inputs["attention_mask"].dtype,
            )
            answer_inputs["input_ids"] = torch.cat([answer_inputs["input_ids"], eos], dim=1)
            answer_inputs["attention_mask"] = torch.cat([answer_inputs["attention_mask"], eos_mask], dim=1)
        answer_input_ids = answer_inputs["input_ids"].to(q_embeds.device)
        answer_attention_mask = answer_inputs["attention_mask"].to(q_embeds.device)

        answer_embeds = self.embedding(answer_input_ids)

        inputs_embeds = torch.cat([q_embeds, z_steps, answer_embeds], dim=1)

        z_mask = torch.ones(
            (batch_size, z_steps.shape[1]),
            device=q_embeds.device,
            dtype=answer_attention_mask.dtype,
        )
        attention_mask = torch.cat([q_attention_mask, z_mask, answer_attention_mask], dim=1)

        labels = torch.cat(
            [
                torch.full(
                    (batch_size, q_embeds.shape[1] + z_steps.shape[1]),
                    -100,
                    device=q_embeds.device,
                    dtype=answer_input_ids.dtype,
                ),
                answer_input_ids,
            ],
            dim=1,
        )
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        position_ids = get_position_ids_from_attention_mask(attention_mask)

        outputs = self.llm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        return outputs.loss



