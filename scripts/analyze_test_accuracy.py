"""
分析 CoT 和 DiffLaR Fused 在测试集上的准确度

从测试结果 JSON 文件中提取准确度信息并对比
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_test_results(json_path: str) -> Tuple[float, int, Dict]:
    """
    加载测试结果并计算准确度
    
    Returns:
        (accuracy, total_samples, detailed_stats)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_samples = len(data)
    correct_samples = 0
    detailed_stats = {
        'correct': [],
        'incorrect': [],
        'total': total_samples,
    }
    
    for idx, sample_data in data.items():
        # 跳过非字典类型的数据
        if not isinstance(sample_data, dict):
            continue
        
        # 每个样本可能有多个预测（列表），取第一个
        acc_list = sample_data.get('acc', [0.0])
        acc = acc_list[0] if isinstance(acc_list, list) else acc_list
        
        if acc == 1.0:
            correct_samples += 1
            detailed_stats['correct'].append({
                'idx': idx,
                'question': sample_data.get('question', '')[:100],  # 截取前100字符
                'answer': sample_data.get('answer', ''),
                'pred_answer': sample_data.get('pred_answer', [''])[0] if isinstance(sample_data.get('pred_answer'), list) else sample_data.get('pred_answer', ''),
            })
        else:
            detailed_stats['incorrect'].append({
                'idx': idx,
                'question': sample_data.get('question', '')[:100],
                'answer': sample_data.get('answer', ''),
                'pred_answer': sample_data.get('pred_answer', [''])[0] if isinstance(sample_data.get('pred_answer'), list) else sample_data.get('pred_answer', ''),
            })
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
    
    return accuracy, total_samples, detailed_stats


def find_test_files(logs_dir: str) -> Dict[str, List[str]]:
    """查找所有测试结果文件"""
    test_files = defaultdict(list)
    
    logs_path = Path(logs_dir)
    
    # 查找 CoT 测试文件
    cot_dir = logs_path / "cot" / "qsa-gsm"
    if cot_dir.exists():
        for exp_dir in cot_dir.iterdir():
            if exp_dir.is_dir():
                for test_file in exp_dir.glob("test_*.json"):
                    test_files['cot'].append(str(test_file))
    
    # 查找 DiffLaR Fused 测试文件
    difflar_dir = logs_path / "difflar_fused" / "qsa-gsm"
    if difflar_dir.exists():
        for exp_dir in difflar_dir.iterdir():
            if exp_dir.is_dir():
                for test_file in exp_dir.glob("test_*.json"):
                    test_files['difflar_fused'].append(str(test_file))
    
    # 按时间排序，最新的在前
    for method in test_files:
        test_files[method].sort(reverse=True)
    
    return test_files


def analyze_all_results(logs_dir: str = "/root/autodl-tmp/colar/logs"):
    """分析所有测试结果"""
    print("="*80)
    print("CoT vs DiffLaR Fused 测试集准确度分析")
    print("="*80)
    
    # 查找测试文件
    test_files = find_test_files(logs_dir)
    
    print(f"\n找到的测试文件:")
    print(f"  CoT: {len(test_files.get('cot', []))} 个文件")
    print(f"  DiffLaR Fused: {len(test_files.get('difflar_fused', []))} 个文件")
    
    results_summary = {}
    
    # 分析 CoT 结果
    if test_files.get('cot'):
        print(f"\n{'='*80}")
        print("CoT 测试结果")
        print("="*80)
        
        cot_results = []
        for test_file in test_files['cot']:
            file_name = Path(test_file).name
            print(f"\n分析文件: {file_name}")
            print(f"  路径: {test_file}")
            
            try:
                accuracy, total_samples, detailed_stats = load_test_results(test_file)
                cot_results.append({
                    'file': file_name,
                    'path': test_file,
                    'accuracy': accuracy,
                    'total_samples': total_samples,
                    'correct': len(detailed_stats['correct']),
                    'incorrect': len(detailed_stats['incorrect']),
                    'detailed_stats': detailed_stats,
                })
                
                print(f"  总样本数: {total_samples}")
                print(f"  正确数: {len(detailed_stats['correct'])}")
                print(f"  错误数: {len(detailed_stats['incorrect'])}")
                print(f"  准确度: {accuracy*100:.2f}%")
                
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        
        if cot_results:
            # 使用最新的结果
            latest_cot = cot_results[0]
            results_summary['cot'] = latest_cot
            print(f"\n{'─'*80}")
            print(f"CoT 最新测试结果 (来自 {latest_cot['file']}):")
            print(f"  准确度: {latest_cot['accuracy']*100:.2f}%")
            print(f"  正确/总数: {latest_cot['correct']}/{latest_cot['total_samples']}")
    
    # 分析 DiffLaR Fused 结果
    if test_files.get('difflar_fused'):
        print(f"\n{'='*80}")
        print("DiffLaR Fused 测试结果")
        print("="*80)
        
        difflar_results = []
        for test_file in test_files['difflar_fused']:
            file_name = Path(test_file).name
            print(f"\n分析文件: {file_name}")
            print(f"  路径: {test_file}")
            
            try:
                accuracy, total_samples, detailed_stats = load_test_results(test_file)
                difflar_results.append({
                    'file': file_name,
                    'path': test_file,
                    'accuracy': accuracy,
                    'total_samples': total_samples,
                    'correct': len(detailed_stats['correct']),
                    'incorrect': len(detailed_stats['incorrect']),
                    'detailed_stats': detailed_stats,
                })
                
                print(f"  总样本数: {total_samples}")
                print(f"  正确数: {len(detailed_stats['correct'])}")
                print(f"  错误数: {len(detailed_stats['incorrect'])}")
                print(f"  准确度: {accuracy*100:.2f}%")
                
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        
        if difflar_results:
            # 使用最新的结果
            latest_difflar = difflar_results[0]
            results_summary['difflar_fused'] = latest_difflar
            print(f"\n{'─'*80}")
            print(f"DiffLaR Fused 最新测试结果 (来自 {latest_difflar['file']}):")
            print(f"  准确度: {latest_difflar['accuracy']*100:.2f}%")
            print(f"  正确/总数: {latest_difflar['correct']}/{latest_difflar['total_samples']}")
    
    # 对比总结
    if results_summary:
        print(f"\n{'='*80}")
        print("准确度对比总结")
        print("="*80)
        
        if 'cot' in results_summary and 'difflar_fused' in results_summary:
            cot_acc = results_summary['cot']['accuracy']
            difflar_acc = results_summary['difflar_fused']['accuracy']
            
            print(f"\nCoT 方法:")
            print(f"  准确度: {cot_acc*100:.2f}%")
            print(f"  正确/总数: {results_summary['cot']['correct']}/{results_summary['cot']['total_samples']}")
            
            print(f"\nDiffLaR Fused 方法:")
            print(f"  准确度: {difflar_acc*100:.2f}%")
            print(f"  正确/总数: {results_summary['difflar_fused']['correct']}/{results_summary['difflar_fused']['total_samples']}")
            
            print(f"\n差异:")
            diff = difflar_acc - cot_acc
            diff_percent = diff * 100
            if diff > 0:
                print(f"  DiffLaR Fused 比 CoT 高 {diff_percent:.2f} 个百分点")
            elif diff < 0:
                print(f"  DiffLaR Fused 比 CoT 低 {abs(diff_percent):.2f} 个百分点")
            else:
                print(f"  两种方法准确度相同")
            
            # 计算相对提升
            if cot_acc > 0:
                relative_improvement = (difflar_acc - cot_acc) / cot_acc * 100
                print(f"  相对变化: {relative_improvement:+.2f}%")
        
        elif 'cot' in results_summary:
            print(f"\n仅找到 CoT 结果:")
            print(f"  准确度: {results_summary['cot']['accuracy']*100:.2f}%")
        
        elif 'difflar_fused' in results_summary:
            print(f"\n仅找到 DiffLaR Fused 结果:")
            print(f"  准确度: {results_summary['difflar_fused']['accuracy']*100:.2f}%")
        
        # 保存结果
        output_file = 'test_accuracy_comparison.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            # 只保存统计信息，不保存详细样本（避免文件过大）
            summary_for_save = {
                'cot': {
                    'file': results_summary['cot']['file'],
                    'accuracy': results_summary['cot']['accuracy'],
                    'total_samples': results_summary['cot']['total_samples'],
                    'correct': results_summary['cot']['correct'],
                    'incorrect': results_summary['cot']['incorrect'],
                } if 'cot' in results_summary else None,
                'difflar_fused': {
                    'file': results_summary['difflar_fused']['file'],
                    'accuracy': results_summary['difflar_fused']['accuracy'],
                    'total_samples': results_summary['difflar_fused']['total_samples'],
                    'correct': results_summary['difflar_fused']['correct'],
                    'incorrect': results_summary['difflar_fused']['incorrect'],
                } if 'difflar_fused' in results_summary else None,
            }
            if 'cot' in results_summary and 'difflar_fused' in results_summary:
                cot_acc = results_summary['cot']['accuracy']
                difflar_acc = results_summary['difflar_fused']['accuracy']
                summary_for_save['comparison'] = {
                    'accuracy_diff': difflar_acc - cot_acc,
                    'accuracy_diff_percent': (difflar_acc - cot_acc) * 100,
                    'relative_improvement_percent': ((difflar_acc - cot_acc) / cot_acc * 100) if cot_acc > 0 else 0,
                }
            
            json.dump(summary_for_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_file}")
        print("="*80)
    
    else:
        print("\n❌ 未找到任何测试结果文件")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 CoT 和 DiffLaR Fused 测试准确度')
    parser.add_argument('--logs_dir', type=str, default='/root/autodl-tmp/colar/logs',
                        help='日志目录路径')
    
    args = parser.parse_args()
    
    analyze_all_results(args.logs_dir)

