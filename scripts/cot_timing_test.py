"""
COT思考时间测量脚本

精确测量COT模型生成思考过程所需的时间
"""

import os
import sys
import time
import torch
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf


def main():
    # 配置
    ckpt_path = "logs/cot/qsa-gsm/20260105-092646_138769/checkpoints/epoch3__step6728__monitor0.412.ckpt"
    workspace_path = "/root/autodl-tmp/colar"
    batch_size = 8
    max_samples = 200  # 测试样本数量
    
    print("="*60)
    print("COT Thinking Time Measurement")
    print("="*60)
    
    # 加载配置
    config = OmegaConf.create({
        'args': {'workspace_path': workspace_path, 'seed': 0},
        'model': {
            'target': 'src.models.cot.LitCot',
            'model_kwargs': {
                'model_id': 'Llama-3.2-1B-Instruct',
                'sft_method': 'cot',
                'chat_template': False,
                'do_lora': True,
                'lora_config': {'r': 128, 'lora_alpha': 32},
                'answer_generation_config': {'max_new_tokens': 256, 'do_sample': False},
                'latent_generation_config': {
                    'max_new_tokens': 256, 
                    'do_sample': False, 
                    'max_n_latent_forward': 256,
                    'compression_factor': 'auto',
                },
            },
            'training_kwargs': {'optimizer': {'target': 'torch.optim.AdamW', 'lr': 0.0001}}
        },
        'dataloader': {'batch_size': batch_size, 'val_batch_size': batch_size, 'num_workers': 4},
        'data_module': {'target': 'src.datasets.qsa.QSADataModule', 'dataset_name': 'gsm'},
    })
    
    # 加载模型
    print("\nLoading model...")
    from src.models.cot import LitCot
    model = LitCot(
        model_kwargs=config.model.model_kwargs,
        training_kwargs=config.model.training_kwargs,
        all_config=config,
    )
    state = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval().cuda()
    print("Model loaded!")
    
    # 加载数据
    print("\nLoading test data...")
    test_data_path = Path(workspace_path) / "datasets" / "text_reasoning" / "gsm" / "test.json"
    with open(test_data_path) as f:
        test_data = json.load(f)
    
    test_samples = [d['question'] for d in test_data[:max_samples]]
    print(f"Loaded {len(test_samples)} test samples")
    
    # 定义生成函数（COT使用自回归生成）
    def cot_generate(questions):
        """COT思考+答案生成"""
        # 准备输入
        suffix = model.speed_template.format('auto') + model.thinking_separator
        input_ids, attention_mask = model.prepare_inputs(
            questions, padding_side="left", part="question", suffix=suffix
        )
        
        # 生成（思考+答案）
        gen_config = {
            'max_new_tokens': 256,
            'do_sample': False,
            'pad_token_id': model.tokenizer.pad_token_id,
            'eos_token_id': model.tokenizer.eos_token_id,
        }
        
        output_ids = model.llm.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            **gen_config
        )
        
        # 计算生成的token数
        input_len = input_ids.shape[1]
        gen_tokens = output_ids.shape[1] - input_len
        return output_ids, gen_tokens
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = cot_generate(test_samples[:2])
    torch.cuda.synchronize()
    
    # 测试
    print(f"\nRunning timing test (batch_size={batch_size})...")
    all_times = []
    all_tokens = []
    
    for i in tqdm(range(0, len(test_samples), batch_size)):
        batch_questions = test_samples[i:i+batch_size]
        if len(batch_questions) < batch_size:
            continue
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _, gen_tokens = cot_generate(batch_questions)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        batch_time = end_time - start_time
        batch_tokens = gen_tokens * len(batch_questions)
        
        all_times.append(batch_time)
        all_tokens.append(batch_tokens)
    
    # 统计结果
    total_time = sum(all_times)
    total_tokens = sum(all_tokens)
    num_samples = len(all_times) * batch_size
    
    avg_time_per_sample = total_time / num_samples
    avg_tokens_per_sample = total_tokens / num_samples
    tokens_per_second = total_tokens / total_time
    
    print("\n" + "="*60)
    print("COT Timing Results")
    print("="*60)
    print(f"Total samples:           {num_samples}")
    print(f"Batch size:              {batch_size}")
    print(f"Total time:              {total_time:.2f}s")
    print(f"")
    print(f"Per-sample metrics:")
    print(f"  Avg thinking time:     {avg_time_per_sample*1000:.1f}ms")
    print(f"  Avg tokens generated:  {avg_tokens_per_sample:.1f} tokens")
    print(f"  Time per token:        {avg_time_per_sample*1000/avg_tokens_per_sample:.2f}ms")
    print(f"")
    print(f"Throughput:")
    print(f"  Tokens/second:         {tokens_per_second:.1f}")
    print(f"  Samples/second:        {num_samples/total_time:.2f}")
    print("="*60)
    
    # 保存结果
    results = {
        'model': 'COT',
        'ckpt_path': ckpt_path,
        'batch_size': batch_size,
        'num_samples': num_samples,
        'total_time_seconds': total_time,
        'avg_thinking_time_ms': avg_time_per_sample * 1000,
        'avg_tokens_per_sample': avg_tokens_per_sample,
        'time_per_token_ms': avg_time_per_sample * 1000 / avg_tokens_per_sample,
        'tokens_per_second': tokens_per_second,
        'samples_per_second': num_samples / total_time,
    }
    
    with open('cot_timing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to cot_timing_results.json")


if __name__ == "__main__":
    main()
