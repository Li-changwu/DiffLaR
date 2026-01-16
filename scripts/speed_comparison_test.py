"""
DiffLaR Fused vs CoT 速度对比测试脚本

对比两种方法在相同数据上的推理速度：
1. CoT: 自回归生成思考步骤（逐token）
2. DiffLaR Fused: 扩散模型并行生成思考步骤
"""

import os
import sys
import time
import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf


class SpeedComparison:
    def __init__(
        self,
        cot_ckpt_path: str,
        difflar_ckpt_path: str,
        workspace_path: str = "/root/autodl-tmp/colar",
        batch_size: int = 8,
        max_samples: int = 200,
        num_inference_steps: int = 128,  # DiffLaR Fused 的推理步数
    ):
        self.workspace_path = workspace_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.num_inference_steps = num_inference_steps
        
        print("="*80)
        print("DiffLaR Fused vs CoT 速度对比测试")
        print("="*80)
        
        # 加载模型
        print("\n[1/3] 加载模型...")
        self.cot_model = self._load_cot_model(cot_ckpt_path)
        self.difflar_model = self._load_difflar_model(difflar_ckpt_path)
        
        # 加载测试数据
        print("\n[2/3] 加载测试数据...")
        self.test_samples = self._load_test_data()
        print(f"已加载 {len(self.test_samples)} 个测试样本")
        
        print("\n[3/3] 准备完成！")
        print("="*80)
    
    def _load_cot_model(self, ckpt_path: str):
        """加载 CoT 模型"""
        print(f"  加载 CoT 模型: {ckpt_path}")
        config = OmegaConf.create({
            'args': {'workspace_path': self.workspace_path, 'seed': 0},
            'model': {
                'target': 'src.models.cot.LitCot',
                'model_kwargs': {
                    'model_id': 'Llama-3.2-1B-Instruct',
                    'sft_method': 'cot',
                    'chat_template': False,
                    'do_lora': True,
                    'lora_config': {'r': 128, 'lora_alpha': 32},
                    'answer_generation_config': {
                        'max_new_tokens': 256,
                        'do_sample': False,
                    },
                    'latent_generation_config': {
                        'max_new_tokens': 256,
                        'do_sample': False,
                        'max_n_latent_forward': 256,
                        'compression_factor': 'auto',
                    },
                },
                'training_kwargs': {'optimizer': {'target': 'torch.optim.AdamW', 'lr': 0.0001}}
            },
            'dataloader': {'batch_size': self.batch_size, 'val_batch_size': self.batch_size, 'num_workers': 4},
            'data_module': {'target': 'src.datasets.qsa.QSADataModule', 'dataset_name': 'gsm'},
        })
        
        from src.models.cot import LitCot
        model = LitCot(
            model_kwargs=config.model.model_kwargs,
            training_kwargs=config.model.training_kwargs,
            all_config=config,
        )
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval().cuda()
        print("  ✓ CoT 模型加载完成")
        return model
    
    def _load_difflar_model(self, ckpt_path: str):
        """加载 DiffLaR Fused 模型"""
        print(f"  加载 DiffLaR Fused 模型: {ckpt_path}")
        config = OmegaConf.create({
            'args': {'workspace_path': self.workspace_path, 'seed': 0},
            'model': {
                'target': 'src.models.difflar_fused.LitDiffLaRFused',
                'model_kwargs': {
                    'model_id': 'Llama-3.2-1B-Instruct',
                    'sft_method': 'difflar_fused',
                    'chat_template': False,
                    'do_lora': True,
                    'lora_config': {'r': 128, 'lora_alpha': 32},
                    'difflar_config': {
                        'max_latent_length': 256,
                        'num_timesteps': 1000,
                        'num_inference_steps': self.num_inference_steps,
                        'train_inference_steps': 10,
                        'noise_schedule': 'linear',
                        'denoiser_layers': 6,
                        'denoiser_heads': 8,
                        'dropout': 0.1,
                        'diffusion_loss_weight': 5.0,
                        'alignment_loss_weight': 4.0,
                        'answer_loss_weight': 1.0,
                        'use_cfg': False,
                        'cfg_scale': 1.5,
                        'normalize_latent': True,
                        'prediction_type': 'flow',
                        'clamp_value': 3.0,
                        'sampler_type': 'euler',
                        'ddim_eta': 0.0,
                        'stage1_epochs': 5,
                        'rollout_start_ratio': 0.3,
                        'rollout_final_ratio': 1.0,
                        'rollout_inference_steps': 10,
                        'stage1_self_cond_prob': 0.5,
                        'stage2_self_cond_prob': 1.0,
                    },
                    'answer_generation_config': {
                        'max_new_tokens': 16,
                        'do_sample': False,
                    },
                },
                'training_kwargs': {'optimizer': {'target': 'torch.optim.AdamW', 'lr': 0.0001}}
            },
            'dataloader': {'batch_size': self.batch_size, 'val_batch_size': self.batch_size, 'num_workers': 4},
            'data_module': {'target': 'src.datasets.qsa.QSADataModule', 'dataset_name': 'gsm'},
        })
        
        from src.models.difflar_fused import LitDiffLaRFused
        model = LitDiffLaRFused(
            model_kwargs=config.model.model_kwargs,
            training_kwargs=config.model.training_kwargs,
            all_config=config,
        )
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval().cuda()
        print("  ✓ DiffLaR Fused 模型加载完成")
        return model
    
    def _load_test_data(self) -> List[str]:
        """加载测试数据"""
        test_data_path = Path(self.workspace_path) / "datasets" / "text_reasoning" / "gsm" / "test.json"
        with open(test_data_path) as f:
            test_data = json.load(f)
        return [d['question'] for d in test_data[:self.max_samples]]
    
    def _test_cot(self) -> Dict:
        """测试 CoT 方法的速度（使用自回归生成思考+答案，逻辑与 cot_timing_test 一致）"""
        print("\n" + "="*80)
        print("测试 CoT 方法（自回归生成）")
        print("="*80)

        # 内部生成函数，参考 scripts/cot_timing_test.py
        def cot_generate(questions: List[str], measure_time: bool = True):
            """
            COT思考+答案生成
            时间测量：从 Query embedding 编码完成后开始，到最终答案生成完成
            """
            # 准备输入（不计时）
            suffix = self.cot_model.speed_template.format("auto") + self.cot_model.thinking_separator
            input_ids, attention_mask = self.cot_model.prepare_inputs(
                questions, padding_side="left", part="question", suffix=suffix
            )
            input_ids = input_ids.to(self.cot_model.device)
            attention_mask = attention_mask.to(self.cot_model.device)
            
            # 编码 Query（不计时，这是输入准备阶段）
            # 实际上 prepare_inputs 已经返回了 token ids，这里只是移动到设备
            
            # 开始计时：从 Query 输入结束（embedding 编码完成）开始
            if measure_time:
                torch.cuda.synchronize()
                start_time = time.perf_counter()

            gen_config = {
                "max_new_tokens": 256,
                "do_sample": False,
                "pad_token_id": self.cot_model.tokenizer.pad_token_id,
                "eos_token_id": self.cot_model.tokenizer.eos_token_id,
            }

            output_ids = self.cot_model.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_config,
            )
            
            if measure_time:
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
            else:
                elapsed_time = None

            # 计算生成的token数
            input_len = input_ids.shape[1]
            gen_tokens = output_ids.shape[1] - input_len
            return output_ids, gen_tokens, elapsed_time

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            _ = cot_generate(self.test_samples[:2], measure_time=False)
        torch.cuda.synchronize()

        # 正式测试
        print(f"运行速度测试 (batch_size={self.batch_size})...")
        print("  测量时间：从 Query 输入结束到最终答案生成完成")
        all_times = []
        all_tokens = []

        for i in tqdm(range(0, len(self.test_samples), self.batch_size)):
            batch_questions = self.test_samples[i : i + self.batch_size]
            if len(batch_questions) < self.batch_size:
                continue

            with torch.no_grad():
                _, gen_tokens, elapsed_time = cot_generate(batch_questions, measure_time=True)

            batch_time = elapsed_time
            batch_tokens = gen_tokens * len(batch_questions)

            all_times.append(batch_time)
            all_tokens.append(batch_tokens)

        # 统计结果
        total_time = sum(all_times)
        total_tokens = sum(all_tokens)
        num_samples = len(all_times) * self.batch_size

        avg_time_per_sample = total_time / num_samples
        avg_tokens_per_sample = total_tokens / num_samples
        tokens_per_second = total_tokens / total_time

        results = {
            "method": "CoT",
            "total_samples": num_samples,
            "batch_size": self.batch_size,
            "total_time_seconds": total_time,
            "avg_time_per_sample_ms": avg_time_per_sample * 1000,
            "avg_tokens_per_sample": avg_tokens_per_sample,
            "time_per_token_ms": avg_time_per_sample * 1000 / avg_tokens_per_sample
            if avg_tokens_per_sample > 0
            else 0,
            "tokens_per_second": tokens_per_second,
            "samples_per_second": num_samples / total_time,
        }

        self._print_results(results)
        return results
    
    def _test_difflar(self) -> Dict:
        """测试 DiffLaR Fused 方法的速度"""
        print("\n" + "="*80)
        print(f"测试 DiffLaR Fused 方法（扩散模型，{self.num_inference_steps}步）")
        print("="*80)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            _ = self.difflar_model.latent_generate(self.test_samples[:2])
        torch.cuda.synchronize()
        
        # 测试
        print(f"运行速度测试 (batch_size={self.batch_size})...")
        print("  测量时间：从 Query 输入结束到最终答案生成完成")
        all_times = []
        all_tokens = []
        diffusion_times = []
        answer_times = []
        
        for i in tqdm(range(0, len(self.test_samples), self.batch_size)):
            batch_questions = self.test_samples[i:i+self.batch_size]
            if len(batch_questions) < self.batch_size:
                continue
            
            # 准备输入（不计时，这是输入准备阶段）
            with torch.no_grad():
                question_input_ids, question_attention_mask = self.difflar_model.prepare_inputs(
                    batch_questions,
                    padding_side="left",
                    part="question",
                    suffix=self.difflar_model.speed_template.format("auto") + self.difflar_model.thinking_separator,
                )
                query_embedding = self.difflar_model.embedding(question_input_ids)
                query_mask = question_attention_mask.float()
            
            # 开始计时：从 Query embedding 编码完成后开始
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # 分别测量扩散生成和答案生成的时间
            with torch.no_grad():
                # 1. 扩散生成思考步骤
                torch.cuda.synchronize()
                diff_start = time.perf_counter()
                if self.difflar_model.use_cfg:
                    steps_embeds = self.difflar_model.latent_diffusion.generate_with_cfg(
                        condition=query_embedding,
                        num_inference_steps=self.num_inference_steps,
                        latent_length=self.difflar_model.max_latent_length,
                        cfg_scale=self.difflar_model.cfg_scale,
                        condition_mask=query_mask,
                        clamp_value=self.difflar_model.clamp_value,
                        use_self_cond=True,
                    )
                else:
                    steps_embeds = self.difflar_model.latent_diffusion.generate(
                        condition=query_embedding,
                        num_inference_steps=self.num_inference_steps,
                        latent_length=self.difflar_model.max_latent_length,
                        condition_mask=query_mask,
                        clamp_value=self.difflar_model.clamp_value,
                        use_self_cond=True,
                    )
                torch.cuda.synchronize()
                diff_end = time.perf_counter()
                diffusion_time = diff_end - diff_start
                
                # 2. 生成答案（拼接并生成）
                ans_start = time.perf_counter()
                sep_text = [self.difflar_model.thinking_separator] * len(batch_questions)
                sep_inputs = self.difflar_model.tokenizer(
                    sep_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.difflar_model.device)
                sep_embeds = self.difflar_model.embedding(sep_inputs.input_ids)
                sep_length = sep_embeds.shape[1]
                
                all_embeds = torch.cat([query_embedding, steps_embeds, sep_embeds], dim=1)
                all_attention_mask = torch.cat(
                    [
                        question_attention_mask,
                        torch.ones(len(batch_questions), self.difflar_model.max_latent_length, 
                                 device=self.difflar_model.device, dtype=question_attention_mask.dtype),
                        torch.ones(len(batch_questions), sep_length, 
                                 device=self.difflar_model.device, dtype=question_attention_mask.dtype),
                    ],
                    dim=1,
                )
                
                answer_generation_config = self.difflar_model.model_kwargs.answer_generation_config
                pred_ids = self.difflar_model.llm.generate(
                    inputs_embeds=all_embeds,
                    attention_mask=all_attention_mask,
                    **answer_generation_config,
                )
                torch.cuda.synchronize()
                ans_end = time.perf_counter()
                answer_time = ans_end - ans_start
                
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            input_len = len(batch_questions[0])  # 近似值
            gen_tokens = pred_ids.shape[1] - input_len
            batch_tokens = gen_tokens * len(batch_questions)
            
            all_times.append(batch_time)
            all_tokens.append(batch_tokens)
            diffusion_times.append(diffusion_time)
            answer_times.append(answer_time)
        
        # 统计结果
        total_time = sum(all_times)
        total_diffusion_time = sum(diffusion_times)
        total_answer_time = sum(answer_times)
        total_tokens = sum(all_tokens)
        num_samples = len(all_times) * self.batch_size
        
        avg_time_per_sample = total_time / num_samples
        avg_diffusion_time = total_diffusion_time / num_samples
        avg_answer_time = total_answer_time / num_samples
        avg_tokens_per_sample = total_tokens / num_samples
        tokens_per_second = total_tokens / total_time
        
        results = {
            'method': 'DiffLaR Fused',
            'num_inference_steps': self.num_inference_steps,
            'total_samples': num_samples,
            'batch_size': self.batch_size,
            'total_time_seconds': total_time,
            'total_diffusion_time_seconds': total_diffusion_time,
            'total_answer_time_seconds': total_answer_time,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'avg_diffusion_time_ms': avg_diffusion_time * 1000,
            'avg_answer_time_ms': avg_answer_time * 1000,
            'avg_tokens_per_sample': avg_tokens_per_sample,
            'time_per_token_ms': avg_time_per_sample * 1000 / avg_tokens_per_sample if avg_tokens_per_sample > 0 else 0,
            'tokens_per_second': tokens_per_second,
            'samples_per_second': num_samples / total_time,
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results: Dict):
        """打印结果"""
        method = results['method']
        print(f"\n{method} 测试结果:")
        print(f"  时间测量范围:          从 Query 输入结束到最终答案生成完成")
        print(f"  总样本数:              {results['total_samples']}")
        print(f"  批次大小:              {results['batch_size']}")
        print(f"  总时间:                {results['total_time_seconds']:.2f}s")
        
        if 'total_diffusion_time_seconds' in results:
            print(f"  ├─ 扩散生成时间:        {results['total_diffusion_time_seconds']:.2f}s ({results['total_diffusion_time_seconds']/results['total_time_seconds']*100:.1f}%)")
            print(f"  └─ 答案生成时间:        {results['total_answer_time_seconds']:.2f}s ({results['total_answer_time_seconds']/results['total_time_seconds']*100:.1f}%)")
        
        print(f"\n  单样本指标:")
        print(f"    平均时间:             {results['avg_time_per_sample_ms']:.1f}ms")
        if 'avg_diffusion_time_ms' in results:
            print(f"    ├─ 扩散生成:          {results['avg_diffusion_time_ms']:.1f}ms")
            print(f"    └─ 答案生成:          {results['avg_answer_time_ms']:.1f}ms")
        print(f"    平均生成token数:       {results['avg_tokens_per_sample']:.1f}")
        print(f"    每token时间:           {results['time_per_token_ms']:.2f}ms")
        
        print(f"\n  吞吐量:")
        print(f"    Tokens/秒:            {results['tokens_per_second']:.1f}")
        print(f"    样本/秒:               {results['samples_per_second']:.2f}")
    
    def compare(self) -> Dict:
        """运行对比测试"""
        cot_results = self._test_cot()
        difflar_results = self._test_difflar()
        
        # 计算加速比
        speedup = cot_results['avg_time_per_sample_ms'] / difflar_results['avg_time_per_sample_ms']
        
        print("\n" + "="*80)
        print("速度对比总结")
        print("="*80)
        print(f"\nCoT 方法:")
        print(f"  平均时间: {cot_results['avg_time_per_sample_ms']:.1f}ms")
        print(f"  吞吐量:   {cot_results['samples_per_second']:.2f} 样本/秒")
        
        print(f"\nDiffLaR Fused 方法 (推理步数={self.num_inference_steps}):")
        print(f"  平均时间: {difflar_results['avg_time_per_sample_ms']:.1f}ms")
        print(f"  吞吐量:   {difflar_results['samples_per_second']:.2f} 样本/秒")
        if 'avg_diffusion_time_ms' in difflar_results:
            print(f"  ├─ 扩散生成: {difflar_results['avg_diffusion_time_ms']:.1f}ms")
            print(f"  └─ 答案生成: {difflar_results['avg_answer_time_ms']:.1f}ms")
        
        print(f"\n加速比:")
        print(f"  DiffLaR Fused 比 CoT 快 {speedup:.2f}x")
        print(f"  时间减少: {(1 - 1/speedup)*100:.1f}%")
        
        # 保存结果
        comparison_results = {
            'cot': cot_results,
            'difflar_fused': difflar_results,
            'speedup': speedup,
            'time_reduction_percent': (1 - 1/speedup) * 100,
        }
        
        output_file = 'speed_comparison_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
        print("="*80)
        
        return comparison_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DiffLaR Fused vs CoT 速度对比测试')
    parser.add_argument('--cot_ckpt', type=str, required=True,
                        help='CoT 模型检查点路径')
    parser.add_argument('--difflar_ckpt', type=str, required=True,
                        help='DiffLaR Fused 模型检查点路径')
    parser.add_argument('--workspace', type=str, default='/root/autodl-tmp/colar',
                        help='工作空间路径')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='最大测试样本数')
    parser.add_argument('--num_inference_steps', type=int, default=128,
                        help='DiffLaR Fused 推理步数')
    
    args = parser.parse_args()
    
    comparator = SpeedComparison(
        cot_ckpt_path=args.cot_ckpt,
        difflar_ckpt_path=args.difflar_ckpt,
        workspace_path=args.workspace,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_inference_steps=args.num_inference_steps,
    )
    
    comparator.compare()


if __name__ == "__main__":
    main()

