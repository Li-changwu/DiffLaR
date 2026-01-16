"""
基础大模型在test数据集上的速度和准确度测试脚本

测试基础模型（未经过CoT或DiffLaR训练）在test数据集上的表现：
1. 速度：生成答案的平均时间
2. 准确度：答案正确率
"""

import os
import sys
import time
import torch
import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaselineModelTester:
    def __init__(
        self,
        model_id: str = "Llama-3.2-1B-Instruct",
        workspace_path: str = "/root/autodl-tmp/colar",
        batch_size: int = 8,
        max_samples: int = 200,
        max_new_tokens: int = 256,
    ):
        self.workspace_path = workspace_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        
        print("="*80)
        print("基础大模型速度和准确度测试")
        print("="*80)
        
        # 加载模型
        print("\n[1/3] 加载模型...")
        self.model, self.tokenizer = self._load_baseline_model(model_id)
        
        # 加载测试数据
        print("\n[2/3] 加载测试数据...")
        self.test_samples = self._load_test_data()
        print(f"已加载 {len(self.test_samples)} 个测试样本")
        
        print("\n[3/3] 准备完成！")
        print("="*80)
    
    def _load_baseline_model(self, model_id: str):
        """加载基础大模型（未经过训练的模型）"""
        print(f"  加载基础模型: {model_id}")
        llm_path = Path(self.workspace_path) / "models" / "llms" / model_id
        
        # 加载tokenizer
        # 对于decoder-only模型，需要使用left padding
        tokenizer = AutoTokenizer.from_pretrained(str(llm_path), padding_side="left")
        # 使用tokenizer已有的pad_token，如果没有则使用eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        print("  ✓ 基础模型加载完成")
        return model, tokenizer
    
    def _load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        test_data_path = Path(self.workspace_path) / "datasets" / "text_reasoning" / "gsm" / "test.json"
        with open(test_data_path) as f:
            test_data = json.load(f)
        return test_data[:self.max_samples]
    
    def _create_prompt(self, question: str, use_chat_template: bool = True, use_few_shot: bool = True) -> str:
        """创建prompt，让模型直接生成答案（确保输出标准格式Answer:XXX）"""
        # 添加few-shot示例来引导模型输出正确格式
        # 注意：few-shot示例只显示问题和答案，不包含思考过程，引导模型直接输出答案
        few_shot_examples = """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: 3

"""
        
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            # 尝试使用chat template（适用于instruct模型）
            try:
                # 构建包含few-shot示例的prompt
                if use_few_shot:
                    user_content = f"{few_shot_examples}Question: {question}\nAnswer:"
                else:
                    user_content = f"Question: {question}\nAnswer:"
                
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that solves math word problems. Provide only the final answer in the format 'Answer: [number]' without showing your reasoning process."
                    },
                    {"role": "user", "content": user_content}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                # 如果chat template不可用，使用简单格式
                print(f"  Warning: Chat template not available, using simple format: {e}")
        
        # 使用简单的prompt格式
        if use_few_shot:
            prompt = f"{few_shot_examples}Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        return prompt
    
    def _extract_answer(self, output_string: str) -> str:
        """从输出中提取答案，确保能处理标准格式Answer:XXX"""
        # 首先尝试提取所有"Answer:"后面的内容，取最后一个（通常是最完整的答案）
        # 支持多种格式：Answer: 18, Answer:18, Answer: 18., Answer:18.等
        patterns = [
            r"Answer:\s*([^\n#]+)",  # Answer: 后面到换行或#之前的内容
            r"answer:\s*([^\n#]+)",  # 小写格式
            r"Answer\s*:\s*([^\n#]+)",
            r"answer\s*:\s*([^\n#]+)",
        ]
        
        # 找到所有匹配的位置，只取最后一个（避免提取few-shot示例中的答案）
        all_matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, output_string, re.IGNORECASE):
                all_matches.append((match.start(), match.group(1).strip()))
        
        if all_matches:
            # 按位置排序，取最后一个（通常是模型生成的答案，而不是few-shot示例）
            all_matches.sort(key=lambda x: x[0])
            answer = all_matches[-1][1]
            # 移除可能的标点符号和特殊字符
            answer = answer.rstrip(".,!?;").strip()
            # 如果答案包含###分隔符，只取###之前的部分
            if "###" in answer:
                answer = answer.split("###")[0].strip()
            # 移除"assistant"等标记
            answer = re.sub(r'\bassistant\b', '', answer, flags=re.IGNORECASE).strip()
            # 尝试提取数字（GSM数据集答案通常是数字，支持逗号分隔）
            # 先移除逗号，然后提取数字
            answer_no_comma = answer.replace(',', '')
            numbers = re.findall(r'-?\d+\.?\d*', answer_no_comma)
            if numbers:
                # 返回最后一个数字（通常是最完整的答案）
                return numbers[-1]
            # 如果没有数字，返回清理后的文本
            if answer:
                return answer
        
        # 如果没有找到Answer标记，尝试从整个输出中提取数字
        # 优先查找最后几行中的数字
        lines = output_string.strip().split("\n")
        if lines:
            # 从后往前查找包含数字的行
            for line in reversed(lines[-5:]):  # 只检查最后5行
                line = line.strip()
                if not line:
                    continue
                # 跳过明显的prompt部分
                if line.startswith("Question:") or line.startswith("Answer:"):
                    continue
                # 移除"assistant"等标记
                line = re.sub(r'\bassistant\b', '', line, flags=re.IGNORECASE).strip()
                # 尝试提取数字（支持逗号）
                line_no_comma = line.replace(',', '')
                numbers = re.findall(r'-?\d+\.?\d*', line_no_comma)
                if numbers:
                    return numbers[-1]
        
        # 如果还是没找到，尝试从整个字符串中提取最后一个数字（支持逗号）
        output_no_comma = output_string.replace(',', '')
        all_numbers = re.findall(r'-?\d+\.?\d*', output_no_comma)
        if all_numbers:
            return all_numbers[-1]
        
        # 最后返回清理后的输出
        return output_string.strip().split("\n")[-1].strip() if output_string.strip() else ""
    
    def _verify_answer(self, gt_answer: str, pred_answer: str) -> float:
        """验证答案是否正确"""
        def get_pure_string(s: str):
            # 移除所有标点符号、空格、换行，并转换为小写
            s = s.strip("#\n ").rstrip(".,!?;").replace(",", "").replace(" ", "").lower()
            # 移除常见的非数字字符
            s = re.sub(r'[^\d\.\-]', '', s)
            return s
        
        gt_answer = get_pure_string(gt_answer)
        pred_answer = get_pure_string(pred_answer)
        
        # 如果都为空，返回False
        if not gt_answer or not pred_answer:
            return 0.0
        
        try:  # 尝试转换为数字进行比较
            gt_answer = float(gt_answer)
            pred_answer = float(pred_answer)
            return float(gt_answer == pred_answer)
        except ValueError:
            # 如果无法转换为数字，进行字符串比较
            return float(gt_answer == pred_answer)
    
    def test(self, debug_prompt: bool = False) -> Dict:
        """运行测试"""
        print("\n" + "="*80)
        print("开始测试基础模型")
        print("="*80)
        
        # 调试：显示第一个样本的prompt格式
        if debug_prompt:
            print("\n[调试] 第一个样本的prompt格式:")
            print("-" * 80)
            sample_prompt = self._create_prompt(self.test_samples[0]['question'])
            print(sample_prompt)
            print("-" * 80)
            print(f"Prompt长度: {len(sample_prompt)} 字符")
            print()
        
        # Warmup
        print("Warming up...")
        warmup_questions = [self._create_prompt(self.test_samples[0]['question'])]
        with torch.no_grad():
            inputs = self.tokenizer(
                warmup_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            _ = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        
        # 测试
        print(f"运行测试 (batch_size={self.batch_size})...")
        all_times = []
        all_tokens = []
        all_acc = []
        all_results = []
        
        for i in tqdm(range(0, len(self.test_samples), self.batch_size)):
            batch_samples = self.test_samples[i:i+self.batch_size]
            if len(batch_samples) < self.batch_size:
                continue
            
            batch_questions = [self._create_prompt(s['question']) for s in batch_samples]
            batch_answers = [s['answer'] for s in batch_samples]
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # 准备输入（使用left padding）
                inputs = self.tokenizer(
                    batch_questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)
                
                # 生成答案
                pred_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            
            # 解码输出
            input_lengths = inputs['attention_mask'].sum(dim=1)
            input_ids = inputs['input_ids']
            
            # 只解码新生成的部分（不包括输入）
            generated_ids = []
            for j in range(len(batch_questions)):
                input_len = input_lengths[j].item()
                # 只取生成的部分（跳过输入部分）
                gen_ids = pred_ids[j][input_len:]
                generated_ids.append(gen_ids)
            
            # 解码生成的文本
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # 也解码完整输出用于调试
            full_output_strings = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            
            # 计算生成的token数和准确度
            batch_tokens = 0
            batch_acc = []
            
            for j, (gen_text, full_output, gt_answer, input_len) in enumerate(zip(generated_texts, full_output_strings, batch_answers, input_lengths)):
                # 计算生成的token数
                actual_input_len = input_len.item()
                gen_tokens = max(0, pred_ids[j].shape[0] - actual_input_len)
                batch_tokens += gen_tokens
                
                # 从生成的文本中提取答案（只使用新生成的部分）
                pred_answer = self._extract_answer(gen_text)
                acc = self._verify_answer(gt_answer, pred_answer)
                batch_acc.append(acc)
                
                # 保存结果
                all_results.append({
                    'question': batch_samples[j]['question'],
                    'ground_truth': gt_answer,
                    'predicted': pred_answer,
                    'generated_text': gen_text,  # 只保存生成的部分
                    'full_output': full_output,  # 保存完整输出用于调试
                    'accuracy': acc,
                })
            
            all_times.append(batch_time)
            all_tokens.append(batch_tokens)
            all_acc.extend(batch_acc)
        
        # 统计结果
        total_time = sum(all_times)
        total_tokens = sum(all_tokens)
        num_samples = len(all_acc)
        
        avg_time_per_sample = total_time / num_samples
        avg_tokens_per_sample = total_tokens / num_samples
        accuracy = sum(all_acc) / len(all_acc) if all_acc else 0.0
        tokens_per_second = total_tokens / total_time
        samples_per_second = num_samples / total_time
        
        results = {
            'method': 'Baseline Model',
            'model_id': self.model_id if hasattr(self, 'model_id') else 'Unknown',
            'total_samples': num_samples,
            'batch_size': self.batch_size,
            'total_time_seconds': total_time,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'avg_tokens_per_sample': avg_tokens_per_sample,
            'time_per_token_ms': avg_time_per_sample * 1000 / avg_tokens_per_sample if avg_tokens_per_sample > 0 else 0,
            'tokens_per_second': tokens_per_second,
            'samples_per_second': samples_per_second,
            'accuracy': accuracy,
            'correct_samples': sum(all_acc),
            'total_samples': len(all_acc),
        }
        
        self._print_results(results)
        return results, all_results
    
    def _print_results(self, results: Dict):
        """打印结果"""
        print(f"\n基础模型测试结果:")
        print(f"  总样本数:              {results['total_samples']}")
        print(f"  批次大小:              {results['batch_size']}")
        print(f"  总时间:                {results['total_time_seconds']:.2f}s")
        
        print(f"\n  单样本指标:")
        print(f"    平均时间:             {results['avg_time_per_sample_ms']:.1f}ms")
        print(f"    平均生成token数:       {results['avg_tokens_per_sample']:.1f}")
        print(f"    每token时间:           {results['time_per_token_ms']:.2f}ms")
        
        print(f"\n  吞吐量:")
        print(f"    Tokens/秒:            {results['tokens_per_second']:.1f}")
        print(f"    样本/秒:               {results['samples_per_second']:.2f}")
        
        print(f"\n  准确度:")
        print(f"    准确率:               {results['accuracy']*100:.2f}%")
        print(f"    正确样本数:           {results['correct_samples']}/{results['total_samples']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='基础大模型速度和准确度测试')
    parser.add_argument('--model_id', type=str, default='Llama-3.2-1B-Instruct',
                        help='模型ID')
    parser.add_argument('--workspace', type=str, default='/root/autodl-tmp/colar',
                        help='工作空间路径')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='最大测试样本数')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成token数')
    parser.add_argument('--output_file', type=str, default='baseline_model_test_results.json',
                        help='结果输出文件')
    parser.add_argument('--debug_prompt', action='store_true',
                        help='显示第一个样本的prompt格式（用于调试）')
    
    args = parser.parse_args()
    
    tester = BaselineModelTester(
        model_id=args.model_id,
        workspace_path=args.workspace,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )
    tester.model_id = args.model_id
    
    results, all_results = tester.test(debug_prompt=args.debug_prompt)
    
    # 保存结果
    output_data = {
        'summary': results,
        'detailed_results': all_results,
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

