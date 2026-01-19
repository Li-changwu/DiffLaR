"""
提取并保存 DiffLaR Hidden Stage1 所需的离线特征。

- Question: 保存 Query Embeddings（作为 Diffusion condition）
- Steps: 保存 Last Hidden States（作为 Diffusion target）

用于 DiffLaR Hidden 算法的数据预处理阶段。
一次性提取 Question Embeddings 和 Steps 的 Last Hidden States，供 Stage1 训练使用。
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


class QuestionStepsAnswerDataset(Dataset):
    """简单的数据集类，用于加载原始数据"""
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "steps": "\n".join(item["steps"]) if isinstance(item["steps"], list) else item["steps"],
            "answer": item["answer"],
        }


def extract_hidden_states(
    model_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    max_question_length: int = 512,
    max_steps_length: int = 512,
    device: str = "cuda",
    hidden_dtype: str = "float16",
):
    """
    提取并保存 Last Hidden States
    
    Args:
        model_path: LLM 模型路径
        data_path: 训练数据路径（JSON格式）
        output_dir: 输出目录
        batch_size: 批处理大小
        max_question_length: Question 最大长度
        max_steps_length: Steps 最大长度
        device: 设备（cuda/cpu）
        hidden_dtype: 保存 hidden states 的 dtype（"float16" 或 "float32"），默认 float16 以节省磁盘
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # 确保 tokenizer 有 pad_token（否则 padding=True 会报错）
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            # 某些模型可能不支持resize，这里尽量不中断
            pass
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Loading data from {data_path}...")
    dataset = QuestionStepsAnswerDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 初始化存储容器
    all_question_embeds = []
    all_steps_hidden = []
    all_question_mask = []
    all_steps_mask = []
    
    print("Extracting hidden states...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            questions = batch["question"]
            steps = batch["steps"]
            
            # 处理 Question
            # 重要：必须使用 padding="max_length" 保证不同 batch 的张量长度一致，
            # 否则 torch.cat 会因为 seq_len 不同而失败
            question_inputs = tokenizer(
                questions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_question_length,
            ).to(device)
            
            question_embeds = model.get_input_embeddings()(question_inputs.input_ids)
            # 参照 CoLaR：将 PAD 位置的 embedding 置零（重要！）
            # 这样即使 padding token 的 embedding 非零，也不会污染 hidden states 分布
            question_embeds = question_embeds * question_inputs.attention_mask.unsqueeze(-1)
            
            # Question 作为 Diffusion condition：直接保存 Query Embeddings（而不是 last hidden state）
            question_mask = question_inputs.attention_mask  # [B, L_q]
            
            # 处理 Steps
            steps_inputs = tokenizer(
                steps,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_steps_length,
            ).to(device)
            
            steps_embeds = model.get_input_embeddings()(steps_inputs.input_ids)
            # 参照 CoLaR：将 PAD 位置的 embedding 置零（重要！）
            steps_embeds = steps_embeds * steps_inputs.attention_mask.unsqueeze(-1)
            
            steps_outputs = model.forward(
                inputs_embeds=steps_embeds,
                attention_mask=steps_inputs.attention_mask,
                output_hidden_states=True,
            )
            
            steps_hidden = steps_outputs.hidden_states[-1]  # [B, L_s, H]
            steps_mask = steps_inputs.attention_mask  # [B, L_s]
            
            # 保存到 CPU（节省显存）
            all_question_embeds.append(question_embeds.cpu())  # 保存 question_embeds 而非 last hidden state
            all_steps_hidden.append(steps_hidden.cpu())
            all_question_mask.append(question_mask.cpu())
            all_steps_mask.append(steps_mask.cpu())
    
    # 合并并保存
    print("Concatenating and saving...")
    all_question_embeds = torch.cat(all_question_embeds, dim=0)  # [N, L_q, H]
    all_steps_hidden = torch.cat(all_steps_hidden, dim=0)  # [N, L_s, H]
    all_question_mask = torch.cat(all_question_mask, dim=0)  # [N, L_q]
    all_steps_mask = torch.cat(all_steps_mask, dim=0)  # [N, L_s]

    # 保存为更省空间的 dtype（mask 不变）
    hidden_dtype = str(hidden_dtype).lower()
    if hidden_dtype in ("fp16", "float16", "half"):
        all_question_embeds = all_question_embeds.to(torch.float16)
        all_steps_hidden = all_steps_hidden.to(torch.float16)
    elif hidden_dtype in ("fp32", "float32", "float"):
        all_question_embeds = all_question_embeds.to(torch.float32)
        all_steps_hidden = all_steps_hidden.to(torch.float32)
    else:
        raise ValueError(f"Unsupported hidden_dtype: {hidden_dtype}. Use float16/float32.")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到文件
    # Stage1: Query Embeddings 作为 Diffusion condition；Steps Last Hidden State 作为 Diffusion target
    torch.save(all_question_embeds, os.path.join(output_dir, "question_embeddings.pt"))
    torch.save(all_steps_hidden, os.path.join(output_dir, "steps_hidden_states.pt"))
    torch.save(all_question_mask, os.path.join(output_dir, "question_attention_mask.pt"))
    torch.save(all_steps_mask, os.path.join(output_dir, "steps_attention_mask.pt"))
    
    # 保存元信息
    metadata = {
        "num_samples": len(dataset),
        "hidden_size": all_question_embeds.shape[-1],
        "max_question_length": all_question_embeds.shape[1],
        "max_steps_length": all_steps_hidden.shape[1],
        "model_path": model_path,
        "question_representation": "input_embeddings_masked",
        "question_files": ["question_embeddings.pt"],
        "steps_representation": "last_hidden_state",
        "steps_files": ["steps_hidden_states.pt"],
        "extraction_config": {
            "batch_size": batch_size,
            "max_question_length": max_question_length,
            "max_steps_length": max_steps_length,
            "hidden_dtype": hidden_dtype,
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Hidden states saved to {output_dir}")
    print(f"Total samples: {metadata['num_samples']}")
    print(f"Hidden size: {metadata['hidden_size']}")
    print(f"Max question length: {metadata['max_question_length']}")
    print(f"Max steps length: {metadata['max_steps_length']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract Last Hidden States for DiffLaR Hidden")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (JSON)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_question_length", type=int, default=512, help="Max question length")
    parser.add_argument("--max_steps_length", type=int, default=512, help="Max steps length")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--hidden_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Dtype for saved hidden states. float16 saves ~50% disk vs float32.",
    )
    
    args = parser.parse_args()
    
    extract_hidden_states(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_question_length=args.max_question_length,
        max_steps_length=args.max_steps_length,
        device=args.device,
        hidden_dtype=args.hidden_dtype,
    )


if __name__ == "__main__":
    main()


