"""
Few-shot Prompting 评估脚本
使用未训练的Qwen3-4B-Base模型 + Few-shot示例 来生成结果
作为baseline与训练后模型对比
"""

import os
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from openai import OpenAI
import pandas as pd
from collections import defaultdict


# Few-shot示例（从训练数据中挑选几个典型例子）
FEW_SHOT_EXAMPLES = [
    {
        "user_prompt": """Java code requirement: You're given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account falls below zero, and at that point function should return True. Otherwise it should return False.
student code: import java.util.*; class Solution { public boolean belowZero(List<Integer> operations) { if (operations.isEmpty()) return true; int balance = 0; for (int op : operations) { balance += op; if (balance < 0) return true; } return false; } }""",
        "feedback": "1) Logical Error: should return false for empty operations, not true."
    },
    {
        "user_prompt": """Java code requirement: Return a greatest common divisor of two integers a and b.
student code: import java.util.*; class Solution { public int gcd(int a, int b) { while (b != 0) { int temp = b; b = a % b; a = temp; } return b; } }""",
        "feedback": "1) Logical Error: should return 'a' instead of 'b' after the loop ends."
    },
    {
        "user_prompt": """Java code requirement: Check if two words have the same characters.
student code: import java.util.*; class Solution { public boolean sameChars(String s0, String s1) { Set<Character> set0 = new HashSet<>(); Set<Character> set1 = new HashSet<>(); for (char c : s0.toCharArray()) set0.add(c); for (char c : s1.toCharArray()) set1.add(c); return set0.equals(set1); } }""",
        "feedback": "No errors found. Code is correct."
    },
    {
        "user_prompt": """Java code requirement: This function takes a list l and returns a list l' such that l' is identical to l in the odd indicies, while its values at the even indicies are equal to the values of the even indicies of l, but sorted.
student code: import java.util.*; class Solution { public List<Integer> sortEven(List<Integer> l) { List<Integer> even = new ArrayList<>(); for (int i = 0; i < l.size(); i += 2) { even.add(l.get(i)); } Collections.sort(even, Collections.reverseOrder()); List<Integer> result = new ArrayList<>(l); for (int i = 0; i < l.size(); i += 2) { result.set(i, even.get(i / 2)); } return result; } }""",
        "feedback": "1) Logical Error: Sorting in descending order instead of ascending order."
    },
]


class FewShotEvaluator:
    """Few-shot Prompting 评估器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Base",
        test_data_path: str = "./data/test_data.json",
        openai_api_key: str = None,
        gpt_model: str = "gpt-4o-mini",
        num_shots: int = 4,  # few-shot示例数量
    ):
        self.model_name = model_name
        self.test_data_path = test_data_path
        self.gpt_model = gpt_model
        self.num_shots = num_shots
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # OpenAI客户端
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"初始化Few-shot评估器...")
        print(f"模型: {model_name}")
        print(f"设备: {self.device}")
        print(f"Few-shot数量: {num_shots}")
        
        # 加载模型和tokenizer
        self.model, self.tokenizer = self.load_model()
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        print(f"✓ 加载 {len(self.test_data)} 个测试样本")
        
        # 构建few-shot prompt
        self.system_prompt = self.build_fewshot_system_prompt()
    
    def load_model(self):
        """加载原始模型（不带LoRA）"""
        print(f"\n加载模型: {self.model_name}")
        
        # 检测精度
        compute_capability = torch.cuda.get_device_capability(0)
        if compute_capability[0] >= 8:
            compute_dtype = torch.bfloat16
            print(f"  使用 bfloat16")
        else:
            compute_dtype = torch.float16
            print(f"  使用 float16")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        model.eval()
        print(f"✓ 模型加载完成")
        
        return model, tokenizer
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        return test_data
    
    def build_fewshot_system_prompt(self) -> str:
        """构建包含few-shot示例的system prompt"""
        
        base_instruction = """Analyze the student's Java code according to the given task requirements.  
Identify all errors, and list them point by point.  

Each error must follow this format:  
1) [Error Type: Syntax Error / Runtime Error / Logical Error] – one-sentence explanation  
2) [Error Type: ...] – one-sentence explanation  
...  

Rules:  
- Each point = exactly one error.  
- Every error must belong to one of: Syntax Error, Runtime Error, Logical Error.  
- If no errors exist, output exactly:  
No errors found. Code is correct.

"""
        
        # 添加few-shot示例
        examples_text = "Here are some examples:\n\n"
        
        for i, example in enumerate(FEW_SHOT_EXAMPLES[:self.num_shots]):
            examples_text += f"--- Example {i+1} ---\n"
            examples_text += f"Input: {example['user_prompt']}\n"
            examples_text += f"Output: {example['feedback']}\n\n"
        
        examples_text += "--- Now analyze the following code ---\n"
        
        return base_instruction + examples_text
    
    def generate_response(self, user_prompt: str) -> str:
        """生成模型响应"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 构建输入
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用Qwen3的thinking模式
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 清理thinking标签
        if "<think>" in response:
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            else:
                response = response.split("<think>")[0].strip()
        
        return response
    
    def gpt_score(self, prediction: str, ground_truth: str, retry: int = 3) -> Dict:
        """使用GPT评分"""
        
        if not prediction or prediction.strip() == "":
            return {
                'type_score': 0.0,
                'count_score': 0.0,
                'content_score': 0.0,
                'total_score': 0.0,
                'reasoning': 'Model prediction is empty'
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
   - 0 points: Completely wrong

3. **Error Content Quality (10 points)**:
   - How accurate and complete are the error descriptions?
   - 10 points: Descriptions are accurate, specific, and match the ground truth closely
   - 7-9 points: Descriptions are mostly accurate with minor missing details
   - 4-6 points: Descriptions capture the main idea but lack precision
   - 1-3 points: Descriptions are vague or partially incorrect
   - 0 points: Descriptions are wrong or missing

Return your evaluation in JSON format:
{{
    "type_score": <float 0-10>,
    "count_score": <float 0-10>,
    "content_score": <float 0-10>,
    "reasoning": "<brief explanation>"
}}

Only return the JSON."""

        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[{"role": "user", "content": scoring_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
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
        
        return {
            'type_score': 0.0,
            'count_score': 0.0,
            'content_score': 0.0,
            'total_score': 0.0,
            'reasoning': 'Scoring failed'
        }
    
    def evaluate(self, max_samples: int = 101) -> List[Dict]:
        """运行评估"""
        print(f"\n{'='*70}")
        print(f"Few-shot Evaluation ({self.num_shots}-shot)")
        print(f"{'='*70}")
        
        results = []
        num_samples = min(len(self.test_data), max_samples)
        
        print(f"\n开始推理和评分 {num_samples} 个测试样本...")
        
        for idx, sample in enumerate(self.test_data[:num_samples]):
            print(f"\r进度: {idx+1}/{num_samples}", end='', flush=True)
            
            # 生成预测
            prediction = self.generate_response(sample['user_prompt'])
            
            # GPT评分
            scores = self.gpt_score(prediction, sample['feedback'])
            
            result = {
                'index': idx,
                'user_prompt': sample['user_prompt'],
                'ground_truth': sample['feedback'],
                'prediction': prediction,
                'scores': scores
            }
            
            results.append(result)
            
            # 避免API限流
            if (idx + 1) % 10 == 0:
                time.sleep(1)
        
        print(f"\n✓ 评估完成")
        
        return results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """计算统计数据"""
        stats = {
            'total_samples': len(results),
            'avg_type_score': 0.0,
            'avg_count_score': 0.0,
            'avg_content_score': 0.0,
            'avg_total_score': 0.0,
            'empty_predictions': 0,
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
    
    def save_results(self, results: List[Dict], output_dir: str = "./evaluation_results_gpt"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        output_file = f"{output_dir}/fewshot_{self.num_shots}shot_gpt_scores.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存详细结果: {output_file}")
        
        # 计算并打印统计
        stats = self.calculate_statistics(results)
        
        print(f"\n{'='*70}")
        print(f"Few-shot ({self.num_shots}-shot) 评估结果")
        print(f"{'='*70}")
        print(f"  样本数: {stats['total_samples']}")
        print(f"  平均错误种类得分: {stats['avg_type_score']:.2f}/10")
        print(f"  平均错误个数得分: {stats['avg_count_score']:.2f}/10")
        print(f"  平均错误内容得分: {stats['avg_content_score']:.2f}/10")
        print(f"  平均总分: {stats['avg_total_score']:.2f}/30 ({stats['avg_total_score']/30*100:.1f}%)")
        print(f"  空预测数: {stats['empty_predictions']}")
        
        return stats
    
    def print_cases(self, results: List[Dict], num_cases: int = 3):
        """打印示例案例"""
        print(f"\n{'='*70}")
        print("示例案例")
        print(f"{'='*70}")
        
        sorted_results = sorted(results, key=lambda x: x['scores']['total_score'])
        
        print(f"\n【低分案例】")
        for idx, result in enumerate(sorted_results[:num_cases]):
            print(f"\n{'─'*70}")
            print(f"案例 {idx+1} - 总分: {result['scores']['total_score']:.1f}/30")
            print(f"{'─'*70}")
            
            print(f"\n【Ground Truth】\n{result['ground_truth']}")
            print(f"\n【Few-shot预测】\n{result['prediction'] if result['prediction'] else '[空输出]'}")
            
            scores = result['scores']
            print(f"\n【评分】种类:{scores['type_score']:.1f} 个数:{scores['count_score']:.1f} 内容:{scores['content_score']:.1f}")
        
        print(f"\n【高分案例】")
        for idx, result in enumerate(sorted_results[-num_cases:]):
            print(f"\n{'─'*70}")
            print(f"案例 {idx+1} - 总分: {result['scores']['total_score']:.1f}/30")
            print(f"{'─'*70}")
            
            print(f"\n【Ground Truth】\n{result['ground_truth']}")
            print(f"\n【Few-shot预测】\n{result['prediction'] if result['prediction'] else '[空输出]'}")


def compare_all_methods(output_dir: str = "./evaluation_results_gpt"):
    """对比所有方法的结果"""
    
    print(f"\n{'='*70}")
    print("方法对比汇总")
    print(f"{'='*70}")
    
    comparison = []
    
    # 读取各方法结果
    methods_files = {
        'FedAvg': 'fedavg_gpt_scores.json',
        'FedAdam': 'fedadam_gpt_scores.json',
        'FedAdam-Flower': 'fedadam_flower_gpt_scores.json',
        'Few-shot (4-shot)': 'fewshot_4shot_gpt_scores.json',
    }
    
    for method_name, filename in methods_files.items():
        filepath = f"{output_dir}/{filename}"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # 计算统计
            type_scores = [r['scores']['type_score'] for r in results]
            count_scores = [r['scores']['count_score'] for r in results]
            content_scores = [r['scores']['content_score'] for r in results]
            total_scores = [r['scores']['total_score'] for r in results]
            
            comparison.append({
                'Method': method_name,
                'Type': f"{sum(type_scores)/len(type_scores):.2f}",
                'Count': f"{sum(count_scores)/len(count_scores):.2f}",
                'Content': f"{sum(content_scores)/len(content_scores):.2f}",
                'Total': f"{sum(total_scores)/len(total_scores):.2f}/30",
                'Percentage': f"{sum(total_scores)/len(total_scores)/30*100:.1f}%"
            })
    
    if comparison:
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        
        # 保存对比表
        df.to_csv(f"{output_dir}/all_methods_comparison.csv", index=False)
        print(f"\n✓ 对比表已保存: {output_dir}/all_methods_comparison.csv")
    else:
        print("未找到任何评估结果文件")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Few-shot Prompting评估')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API Key')
    parser.add_argument('--num_shots', type=int, default=4, help='Few-shot示例数量')
    parser.add_argument('--max_samples', type=int, default=101, help='最大测试样本数')
    parser.add_argument('--compare', action='store_true', help='对比所有方法结果')
    
    args = parser.parse_args()
    
    # 如果只是对比结果
    if args.compare:
        compare_all_methods()
        return
    
    # 运行few-shot评估
    evaluator = FewShotEvaluator(
        model_name="Qwen/Qwen3-4B-Base",
        test_data_path="./data/test_data.json",
        openai_api_key=args.api_key,
        num_shots=args.num_shots,
    )
    
    # 评估
    results = evaluator.evaluate(max_samples=args.max_samples)
    
    # 保存结果
    evaluator.save_results(results)
    
    # 打印案例
    evaluator.print_cases(results)
    
    # 对比所有方法
    print("\n")
    compare_all_methods()


if __name__ == "__main__":
    main()