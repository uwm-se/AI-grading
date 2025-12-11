"""
联邦学习模型评估 - 使用GPT评分（修复版）
通过OpenAI API对模型输出进行自动评分（满分10分）
评分维度：错误种类、错误个数、错误内容
"""

import os
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Tuple
from openai import OpenAI
import pandas as pd
from collections import defaultdict


class GPTScoreEvaluator:
    """基于GPT的模型评估器"""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-4B-Base",
        result_dir: str = "./java_error_federated_results",
        test_data_path: str = "./data/test_data.json",
        openai_api_key: str = None,
        openai_base_url: str = None,
        gpt_model: str = "gpt-4o-mini"
    ):
        self.base_model_name = base_model_name
        self.result_dir = result_dir
        self.test_data_path = test_data_path
        self.gpt_model = gpt_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化OpenAI客户端
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"初始化GPT评分器...")
        print(f"设备: {self.device}")
        print(f"GPT模型: {self.gpt_model}")
        
        # 加载tokenizer - ⭐ 修复1: 设置padding_side和pad_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left"  # 生成时用left padding
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        print(f"✓ 加载 {len(self.test_data)} 个测试样本")
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        return test_data
    
    def load_merged_model(self, method: str):
        """加载并合并LoRA模型"""
        print(f"\n加载 {method.upper()} 模型...")
        
        lora_path = f"{self.result_dir}/{method}/final_model"
        
        if not os.path.exists(lora_path):
            print(f"  ⚠️ 模型路径不存在: {lora_path}")
            return None
        
        # ⭐ 修复2: 检测精度
        compute_capability = torch.cuda.get_device_capability(0)
        if compute_capability[0] >= 8:
            compute_dtype = torch.bfloat16
            print(f"  使用 bfloat16")
        else:
            compute_dtype = torch.float16
            print(f"  使用 float16")
        
        # 加载基础模型
        print(f"  1/3 加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA
        print(f"  2/3 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        # 合并
        print(f"  3/3 合并LoRA与基础模型...")
        merged_model = model.merge_and_unload()
        
        print(f"  ✓ {method.upper()} 模型加载完成")
        
        return merged_model

    def generate_response(self, model, system_prompt: str, user_prompt: str) -> str:
        """生成模型响应 - ⭐ 修复版"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ⭐ 修复3: Qwen3需要设置enable_thinking=False来禁用思考模式
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # ⭐ 关键！禁用Qwen3的thinking模式
            )
        except TypeError:
            # 如果不支持enable_thinking参数，就不传
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,  # ⭐ 修复4: 稍微提高temperature
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,  # ⭐ 修复5: 使用pad_token_id
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 只解码新生成的tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # ⭐ 修复6: 清理可能的thinking标签
        if "<think>" in response:
            # 去掉thinking部分
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            else:
                response = response.split("<think>")[0].strip()
        
        # 调试信息
        if not response:
            print(f"\n⚠️ 警告：模型未生成内容")
            print(f"  输入token数: {input_length}")
            print(f"  输出token数: {len(outputs[0])}")
            print(f"  新生成token数: {len(generated_tokens)}")
            # 打印原始输出看看
            raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(f"  原始输出（前500字符）: {raw_output[:500]}")
            return ""
        
        return response
    
    def gpt_score(self, prediction: str, ground_truth: str, retry: int = 3) -> Dict:
        """
        使用GPT对预测结果进行评分
        """
        if not prediction or prediction.strip() == "":
            return {
                'type_score': 0.0,
                'count_score': 0.0,
                'content_score': 0.0,
                'total_score': 0.0,
                'reasoning': 'Model prediction is empty - no output generated'
            }
        
        scoring_prompt = f"""You are an expert code reviewer evaluating error detection results for Java code.

Ground Truth (Standard Answer):
{ground_truth}

Model Prediction:
{prediction}

Please evaluate the prediction based on three dimensions (each worth 10 points):

1. **Error Type Match (10 points)**:
   - Does the prediction correctly identify error types (Syntax Error, Runtime Error, Logical Error)?
   - 10 points: All error types match perfectly
   - 7-9 points: Most error types match with minor discrepancies
   - 4-6 points: Some error types match but with significant issues
   - 1-3 points: Few error types match
   - 0 points: Completely wrong or opposite

2. **Error Count Match (10 points)**:
   - Does the prediction identify the correct number of errors?
   - 10 points: Exact same number of errors
   - 7-9 points: Off by 1 error
   - 4-6 points: Off by 2 errors
   - 1-3 points: Off by 3+ errors
   - 0 points: Completely wrong (e.g., says "no errors" when there are errors, or vice versa)

3. **Error Content Quality (10 points)**:
   - How accurate and complete are the error descriptions?
   - 10 points: Descriptions are accurate, specific, and match the ground truth closely
   - 7-9 points: Descriptions are mostly accurate with minor missing details
   - 4-6 points: Descriptions capture the main idea but lack precision
   - 1-3 points: Descriptions are vague or partially incorrect
   - 0 points: Descriptions are wrong or missing

**Important Notes**:
- If ground truth says "No errors found. Code is correct." or similar, and prediction also identifies no errors, give full marks.
- Focus on semantic meaning, not exact wording.
- Minor phrasing differences should not heavily penalize scores.

Return your evaluation in the following JSON format:
{{
    "type_score": <float 0-10>,
    "count_score": <float 0-10>,
    "content_score": <float 0-10>,
    "reasoning": "<brief explanation of your scoring>"
}}

Only return the JSON, no additional text."""

        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "user", "content": scoring_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                
                result['total_score'] = (
                    result['type_score'] + 
                    result['count_score'] + 
                    result['content_score']
                )
                
                return result
                
            except Exception as e:
                print(f"  ⚠️ GPT评分失败 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(2)
                else:
                    return {
                        'type_score': 0.0,
                        'count_score': 0.0,
                        'content_score': 0.0,
                        'total_score': 0.0,
                        'reasoning': f"评分失败: {str(e)}"
                    }
    
    def evaluate_model(self, method: str) -> List[Dict]:
        """评估单个模型"""
        print(f"\n{'='*70}")
        print(f"评估 {method.upper()} 模型")
        print(f"{'='*70}")
        
        model = self.load_merged_model(method)
        if model is None:
            return None
        
        model.eval()
        
        results = []
        
        print(f"\n开始推理和评分 {len(self.test_data)} 个测试样本...")
        
        for idx, sample in enumerate(self.test_data):
            if idx == 101:
                break
            print(f"\r进度: {idx+1}/{min(len(self.test_data), 101)}", end='', flush=True)
            
            prediction = self.generate_response(
                model,
                sample['system_prompt'],
                sample['user_prompt']
            )
            
            scores = self.gpt_score(prediction, sample['feedback'])
            
            result = {
                'index': idx,
                'system_prompt': sample['system_prompt'],
                'user_prompt': sample['user_prompt'],
                'ground_truth': sample['feedback'],
                'prediction': prediction,
                'scores': scores
            }
            
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                time.sleep(1)
        
        print(f"\n✓ 推理和评分完成")
        
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """计算统计数据"""
        stats = {
            'total_samples': len(results),
            'avg_type_score': 0.0,
            'avg_count_score': 0.0,
            'avg_content_score': 0.0,
            'avg_total_score': 0.0,
            'empty_predictions': 0,  # ⭐ 新增：统计空预测数
        }
        
        type_scores = []
        count_scores = []
        content_scores = []
        total_scores = []
        
        for result in results:
            scores = result['scores']
            type_scores.append(scores['type_score'])
            count_scores.append(scores['count_score'])
            content_scores.append(scores['content_score'])
            total_scores.append(scores['total_score'])
            
            if not result['prediction'] or result['prediction'].strip() == "":
                stats['empty_predictions'] += 1
        
        stats['avg_type_score'] = sum(type_scores) / len(type_scores) if type_scores else 0
        stats['avg_count_score'] = sum(count_scores) / len(count_scores) if count_scores else 0
        stats['avg_content_score'] = sum(content_scores) / len(content_scores) if content_scores else 0
        stats['avg_total_score'] = sum(total_scores) / len(total_scores) if total_scores else 0
        
        return stats
    
    def print_detailed_cases(self, results: List[Dict], method: str, num_cases: int = 5):
        """打印详细案例"""
        print(f"\n{'='*70}")
        print(f"{method.upper()} - 详细案例分析")
        print(f"{'='*70}")
        
        sorted_results = sorted(results, key=lambda x: x['scores']['total_score'])
        
        print(f"\n【低分案例 - 需要改进】")
        for idx, result in enumerate(sorted_results[:num_cases]):
            print(f"\n{'─'*70}")
            print(f"案例 {idx+1} (样本索引: {result['index']}) - 总分: {result['scores']['total_score']:.1f}/30")
            print(f"{'─'*70}")
            
            user_prompt = result['user_prompt']
            if len(user_prompt) > 200:
                user_prompt = user_prompt[:200] + "..."
            print(f"\n任务: {user_prompt}")
            
            print(f"\n【Ground Truth】")
            print(result['ground_truth'])
            
            print(f"\n【{method.upper()} 预测】")
            print(result['prediction'] if result['prediction'] else "[空输出]")
            
            scores = result['scores']
            print(f"\n【GPT评分】")
            print(f"  错误种类: {scores['type_score']:.1f}/10")
            print(f"  错误个数: {scores['count_score']:.1f}/10")
            print(f"  错误内容: {scores['content_score']:.1f}/10")
            print(f"  总分: {scores['total_score']:.1f}/30")
            print(f"\n  评分理由: {scores['reasoning']}")
        
        print(f"\n{'─'*70}")
        print(f"\n【高分案例 - 表现优秀】")
        for idx, result in enumerate(sorted_results[-num_cases:]):
            print(f"\n{'─'*70}")
            print(f"案例 {idx+1} (样本索引: {result['index']}) - 总分: {result['scores']['total_score']:.1f}/30")
            print(f"{'─'*70}")
            
            print(f"\n【Ground Truth】")
            print(result['ground_truth'])
            
            print(f"\n【{method.upper()} 预测】")
            print(result['prediction'] if result['prediction'] else "[空输出]")
            
            scores = result['scores']
            print(f"\n【GPT评分】")
            print(f"  总分: {scores['total_score']:.1f}/30 (种类:{scores['type_score']:.1f} 个数:{scores['count_score']:.1f} 内容:{scores['content_score']:.1f})")
    
    def save_results(self, all_results: Dict, output_dir: str = "./evaluation_results_gpt"):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        for method, results in all_results.items():
            if results is None:
                continue
            
            output_file = f"{output_dir}/{method}_gpt_scores.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ 保存 {method} 详细结果: {output_file}")
        
        self.generate_comparison_table(all_results, output_dir)
    
    def generate_comparison_table(self, all_results: Dict, output_dir: str):
        """生成对比表格"""
        comparison_data = []
        
        for method, results in all_results.items():
            if results is None:
                continue
            
            stats = self.calculate_statistics(results)
            
            row = {
                'Method': method.upper(),
                'Type Score': f"{stats['avg_type_score']:.2f}/10",
                'Count Score': f"{stats['avg_count_score']:.2f}/10",
                'Content Score': f"{stats['avg_content_score']:.2f}/10",
                'Total Score': f"{stats['avg_total_score']:.2f}/30",
                'Percentage': f"{stats['avg_total_score']/30*100:.1f}%",
                'Empty': stats['empty_predictions']
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        csv_file = f"{output_dir}/model_comparison_gpt.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ 保存对比表格: {csv_file}")
        
        print(f"\n{'='*70}")
        print("GPT评分 - 模型对比总结")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("详细统计")
        print(f"{'='*70}")
        for method, results in all_results.items():
            if results is None:
                continue
            
            stats = self.calculate_statistics(results)
            print(f"\n{method.upper()}:")
            print(f"  平均错误种类得分: {stats['avg_type_score']:.2f}/10")
            print(f"  平均错误个数得分: {stats['avg_count_score']:.2f}/10")
            print(f"  平均错误内容得分: {stats['avg_content_score']:.2f}/10")
            print(f"  平均总分: {stats['avg_total_score']:.2f}/30 ({stats['avg_total_score']/30*100:.1f}%)")
            print(f"  空预测数: {stats['empty_predictions']}/{stats['total_samples']}")
    
    def run_evaluation(self, methods: List[str] = None):
        """运行完整评估"""
        if methods is None:
            methods = ['fedavg', 'fedprox', 'fedadam']
        
        print(f"\n{'='*70}")
        print(f"联邦学习模型评估 - GPT评分")
        print(f"{'='*70}")
        print(f"评估方法: {', '.join([m.upper() for m in methods])}")
        print(f"测试样本: {len(self.test_data)}")
        print(f"GPT模型: {self.gpt_model}")
        
        all_results = {}
        
        for method in methods:
            results = self.evaluate_model(method)
            if results is not None:
                all_results[method] = results
                
                stats = self.calculate_statistics(results)
                
                print(f"\n{method.upper()} 统计:")
                print(f"  错误种类得分: {stats['avg_type_score']:.2f}/10")
                print(f"  错误个数得分: {stats['avg_count_score']:.2f}/10")
                print(f"  错误内容得分: {stats['avg_content_score']:.2f}/10")
                print(f"  平均总分: {stats['avg_total_score']:.2f}/30 ({stats['avg_total_score']/30*100:.1f}%)")
                print(f"  空预测数: {stats['empty_predictions']}")
                
                self.print_detailed_cases(results, method, num_cases=3)
        
        print(f"\n{'='*70}")
        print("保存评估结果")
        print(f"{'='*70}")
        self.save_results(all_results)
        
        return all_results


# ⭐ 新增：快速测试函数
def quick_test(
    base_model_name: str = "Qwen/Qwen3-4B-Base",
    adapter_path: str = "./java_error_federated_results/fedadam_flower/final_model"
):
    """快速测试模型是否能正常生成"""
    
    print("="*60)
    print("快速测试 - 检查模型生成能力")
    print("="*60)
    
    # 检测精度
    compute_capability = torch.cuda.get_device_capability(0)
    if compute_capability[0] >= 8:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    
    print(f"\n1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"2. 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"3. 加载LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print(f"4. 合并模型...")
    model = model.merge_and_unload()
    model.eval()
    
    # 测试prompt
    system_prompt = """Analyze the student's Java code according to the given task requirements.  
Identify all errors, and list them point by point.  

Each error must follow this format:  
1) [Error Type: Syntax Error / Runtime Error / Logical Error] – one-sentence explanation  

Rules:  
- Each point = exactly one error.  
- Every error must belong to one of: Syntax Error, Runtime Error, Logical Error.  
- If no errors exist, output exactly:  
No errors found. Code is correct.
"""
    
    user_prompt = """Java code requirement: Return True if list elements are monotonically increasing or decreasing.
student code: import java.util.*; class Solution { public boolean monotonic(List<Integer> l) { if (l.size() <= 2) return true; int direction = 0; for (int i = 1; i < l.size(); i++) { if (l.get(i) > l.get(i-1)) { if (direction == 0) direction = 1; else if (direction == -1) return false; } else if (l.get(i) < l.get(i-1)) { if (direction == 0) direction = -1; else if (direction == 1) return false; } } return true; } }"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"\n5. 生成测试...")
    
    # 尝试带enable_thinking=False
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        print("   使用 enable_thinking=False")
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("   不支持 enable_thinking 参数")
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"   输入token数: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    print(f"   输出token数: {len(outputs[0])}")
    
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 清理thinking标签
    if "<think>" in response:
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        else:
            response = response.split("<think>")[0].strip()
    
    print(f"\n{'='*60}")
    print("测试结果")
    print(f"{'='*60}")
    
    if response:
        print(f"✅ 生成成功!")
        print(f"\n模型输出:\n{response}")
    else:
        print(f"❌ 生成失败 - 输出为空")
        print(f"\n原始输出:")
        raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(raw[:1000])
    
    return response


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用GPT评分的联邦学习模型评估')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API Key')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o-mini',
                        help='GPT模型名称')
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['fedadam_flower'],
                        help='要评估的方法')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模型生成能力')
    parser.add_argument('--adapter_path', type=str,
                        default='./java_error_federated_results/fedadam_flower/final_model',
                        help='LoRA adapter路径（用于quick_test）')
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick_test:
        quick_test(adapter_path=args.adapter_path)
        return
    
    # 完整评估模式
    evaluator = GPTScoreEvaluator(
        base_model_name="Qwen/Qwen3-4B-Base",
        result_dir="./java_error_federated_results",
        test_data_path="./data/test_data.json",
        openai_api_key=args.api_key,
        gpt_model=args.gpt_model
    )
    
    all_results = evaluator.run_evaluation(methods=args.methods)
    
    print(f"\n{'='*70}")
    print("评估完成！")
    print(f"{'='*70}")
    print("结果已保存到: ./evaluation_results_gpt/")


if __name__ == "__main__":
    main()