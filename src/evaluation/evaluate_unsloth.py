"""
集中式训练模型评估 - 使用GPT评分
评估LoRA微调后的模型在Java代码错误检测任务上的性能

评分维度：
1. 错误种类匹配度 (0-10分)
2. 错误个数匹配度 (0-10分)  
3. 错误内容质量 (0-10分)
总分：0-30分
"""

import os
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List
from openai import OpenAI
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class CentralizedModelEvaluator:
    """集中式训练模型评估器"""
    
    def __init__(
        self,
        base_model_name: str = "unsloth/Qwen3-8B-Base",
        lora_path: str = "./qwen3_java_evaluator_lora/final_model",
        test_data_path: str = "/mnt/user-data/uploads/test_data.json",
        openai_api_key: str = None,
        gpt_model: str = "gpt-4o-mini",
        output_dir: str = "./centralized_evaluation_results"
    ):
        """
        初始化评估器
        
        Args:
            base_model_name: 基座模型名称
            lora_path: LoRA适配器路径
            test_data_path: 测试数据路径
            openai_api_key: OpenAI API密钥
            gpt_model: GPT模型名称
            output_dir: 结果输出目录
        """
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.test_data_path = test_data_path
        self.gpt_model = gpt_model
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化OpenAI客户端
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # 从环境变量读取
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"{'='*70}")
        print(f"集中式训练模型评估器初始化")
        print(f"{'='*70}")
        print(f"设备: {self.device}")
        print(f"基座模型: {base_model_name}")
        print(f"LoRA路径: {lora_path}")
        print(f"GPT模型: {gpt_model}")
        
        # 加载tokenizer
        print(f"\n加载Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer加载完成")
        
        # 加载测试数据
        print(f"\n加载测试数据...")
        self.test_data = self.load_test_data()
        print(f"✓ 加载 {len(self.test_data)} 个测试样本")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
        return test_data
    
    def load_and_merge_model(self):
        """加载并合并LoRA模型"""
        print(f"\n{'='*70}")
        print(f"加载并合并模型")
        print(f"{'='*70}")
        
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"LoRA路径不存在: {self.lora_path}")
        
        # 1. 加载基础模型
        print(f"1/3 加载基座模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ 基座模型加载完成")
        
        # 2. 加载LoRA适配器
        print(f"\n2/3 加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, self.lora_path)
        print(f"✓ LoRA适配器加载完成")
        
        # 3. 合并模型
        print(f"\n3/3 合并LoRA与基座模型...")
        merged_model = model.merge_and_unload()
        print(f"✓ 模型合并完成")
        
        merged_model.eval()
        
        return merged_model
    
    # def generate_response(self, model, system_prompt: str, user_prompt: str) -> str:
    #     """
    #     生成模型响应
        
    #     Args:
    #         model: 模型
    #         system_prompt: 系统提示词
    #         user_prompt: 用户提示词
            
    #     Returns:
    #         模型生成的响应
    #     """
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]
        
    #     # 构建输入
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
        
    #     inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
    #     # 生成
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             **inputs,
    #             max_new_tokens=512,
    #             temperature=0.1,
    #             top_p=0.9,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
        
    #     # 只解码新生成的tokens
    #     input_length = inputs['input_ids'].shape[1]
    #     generated_tokens = outputs[0][input_length:]
    #     response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
    #     # 检查空输出
    #     if not response:
    #         print(f"\n⚠️ 警告：模型未生成内容")
    #         return ""
        
    #     return response

    def generate_response(self, model, system_prompt: str, user_prompt: str) -> str:
        """
        生成模型响应
        
        Args:
            model: 模型
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            模型生成的响应
        """
        # 手动构建 Qwen3 chat 格式（与训练时完全一致）
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        text += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")  # 添加结束符
            )
        
        # 只解码新生成的tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 移除可能的结束标记
        if response.endswith("<|im_end|>"):
            response = response[:-len("<|im_end|>")].strip()
        
        # 检查空输出
        if not response:
            print(f"\n⚠️ 警告：模型未生成内容")
            return ""
        
        return response
    
    def gpt_score(self, prediction: str, ground_truth: str, retry: int = 3) -> Dict:
        """
        使用GPT对预测结果进行评分
        
        Args:
            prediction: 模型预测
            ground_truth: 标准答案
            retry: 重试次数
            
        Returns:
            评分结果字典，包含：
            - type_score: 错误种类得分 (0-10)
            - count_score: 错误个数得分 (0-10)
            - content_score: 错误内容得分 (0-10)
            - total_score: 总分 (0-30)
            - reasoning: 评分理由
        """
        
        # 处理空预测
        if not prediction or prediction.strip() == "":
            return {
                'type_score': 0.0,
                'count_score': 0.0,
                'content_score': 0.0,
                'total_score': 0.0,
                'reasoning': '模型未生成任何输出'
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
                    messages=[{"role": "user", "content": scoring_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                
                # 计算总分
                result['total_score'] = (
                    result['type_score'] + 
                    result['count_score'] + 
                    result['content_score']
                )
                
                return result
                
            except Exception as e:
                print(f"\n⚠️ GPT评分失败 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(2)
                else:
                    # 返回默认分数
                    return {
                        'type_score': 0.0,
                        'count_score': 0.0,
                        'content_score': 0.0,
                        'total_score': 0.0,
                        'reasoning': f"评分失败: {str(e)}"
                    }
    
    def evaluate_model(self, model) -> List[Dict]:
        """
        评估模型
        
        Args:
            model: 已加载的模型
            
        Returns:
            评估结果列表
        """
        print(f"\n{'='*70}")
        print(f"开始评估模型")
        print(f"{'='*70}")
        print(f"测试样本数: {len(self.test_data)}")
        
        results = []
        
        # 使用tqdm显示进度
        for idx, sample in enumerate(tqdm(self.test_data, desc="评估进度")):

            # 生成预测
            prediction = self.generate_response(
                model,
                sample['system_prompt'],
                sample['user_prompt']
            )
            
            # GPT评分
            scores = self.gpt_score(prediction, sample['feedback'])
            
            # 保存结果
            result = {
                'index': idx,
                'system_prompt': sample['system_prompt'],
                'user_prompt': sample['user_prompt'],
                'ground_truth': sample['feedback'],
                'prediction': prediction,
                'scores': scores
            }
            
            results.append(result)
            
            # 每10个样本休息一下，避免API限流
            if (idx + 1) % 10 == 0:
                time.sleep(1)
        
        print(f"\n✓ 评估完成！")
        
        return results
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """
        计算统计数据
        
        Args:
            results: 评估结果列表
            
        Returns:
            统计数据字典
        """
        stats = {
            'total_samples': len(results),
            'avg_type_score': 0.0,
            'avg_count_score': 0.0,
            'avg_content_score': 0.0,
            'avg_total_score': 0.0,
            'std_total_score': 0.0,
            'score_distribution': {
                'type': defaultdict(int),
                'count': defaultdict(int),
                'content': defaultdict(int)
            }
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
            
            # 分数分布（按2分一档统计）
            stats['score_distribution']['type'][int(scores['type_score'] // 2) * 2] += 1
            stats['score_distribution']['count'][int(scores['count_score'] // 2) * 2] += 1
            stats['score_distribution']['content'][int(scores['content_score'] // 2) * 2] += 1
        
        stats['avg_type_score'] = sum(type_scores) / len(type_scores)
        stats['avg_count_score'] = sum(count_scores) / len(count_scores)
        stats['avg_content_score'] = sum(content_scores) / len(content_scores)
        stats['avg_total_score'] = sum(total_scores) / len(total_scores)
        
        # 计算标准差
        mean_total = stats['avg_total_score']
        variance = sum((s - mean_total) ** 2 for s in total_scores) / len(total_scores)
        stats['std_total_score'] = variance ** 0.5
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """打印统计信息"""
        print(f"\n{'='*70}")
        print(f"评估统计")
        print(f"{'='*70}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"\n平均得分:")
        print(f"  错误种类: {stats['avg_type_score']:.2f}/10")
        print(f"  错误个数: {stats['avg_count_score']:.2f}/10")
        print(f"  错误内容: {stats['avg_content_score']:.2f}/10")
        print(f"  总分: {stats['avg_total_score']:.2f}/30 ({stats['avg_total_score']/30*100:.1f}%)")
        print(f"  标准差: {stats['std_total_score']:.2f}")
    
    def print_detailed_cases(self, results: List[Dict], num_low: int = 5, num_high: int = 5):
        """
        打印详细案例
        
        Args:
            results: 评估结果
            num_low: 低分案例数量
            num_high: 高分案例数量
        """
        # 按总分排序
        sorted_results = sorted(results, key=lambda x: x['scores']['total_score'])
        
        # 低分案例
        print(f"\n{'='*70}")
        print(f"低分案例分析（需要改进）")
        print(f"{'='*70}")
        
        for idx, result in enumerate(sorted_results[:num_low]):
            print(f"\n{'-'*70}")
            print(f"案例 {idx+1} (样本 #{result['index']}) - 总分: {result['scores']['total_score']:.1f}/30")
            print(f"{'-'*70}")
            
            # 任务（简化显示）
            user_prompt = result['user_prompt']
            if len(user_prompt) > 150:
                user_prompt = user_prompt[:150] + "..."
            print(f"\n【任务】\n{user_prompt}")
            
            # Ground Truth
            print(f"\n【标准答案】\n{result['ground_truth']}")
            
            # 模型预测
            print(f"\n【模型预测】\n{result['prediction']}")
            
            # 评分
            scores = result['scores']
            print(f"\n【GPT评分】")
            print(f"  错误种类: {scores['type_score']:.1f}/10")
            print(f"  错误个数: {scores['count_score']:.1f}/10")
            print(f"  错误内容: {scores['content_score']:.1f}/10")
            print(f"  总分: {scores['total_score']:.1f}/30")
            print(f"  评分理由: {scores['reasoning']}")
        
        # 高分案例
        print(f"\n{'='*70}")
        print(f"高分案例分析（表现优秀）")
        print(f"{'='*70}")
        
        for idx, result in enumerate(sorted_results[-num_high:]):
            print(f"\n{'-'*70}")
            print(f"案例 {idx+1} (样本 #{result['index']}) - 总分: {result['scores']['total_score']:.1f}/30")
            print(f"{'-'*70}")
            
            print(f"\n【标准答案】\n{result['ground_truth']}")
            print(f"\n【模型预测】\n{result['prediction']}")
            
            scores = result['scores']
            print(f"\n【GPT评分】总分: {scores['total_score']:.1f}/30")
            print(f"  (种类:{scores['type_score']:.1f} 个数:{scores['count_score']:.1f} 内容:{scores['content_score']:.1f})")
    
    def save_results(self, results: List[Dict], stats: Dict):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            stats: 统计数据
        """
        print(f"\n{'='*70}")
        print(f"保存评估结果")
        print(f"{'='*70}")
        
        # 保存详细结果
        results_file = f"{self.output_dir}/detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 详细结果: {results_file}")
        
        # 保存统计数据
        stats_file = f"{self.output_dir}/statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✓ 统计数据: {stats_file}")
        
        # 生成CSV摘要
        summary_data = {
            'Metric': [
                'Error Type Score',
                'Error Count Score',
                'Error Content Score',
                'Total Score',
                'Percentage'
            ],
            'Value': [
                f"{stats['avg_type_score']:.2f}/10",
                f"{stats['avg_count_score']:.2f}/10",
                f"{stats['avg_content_score']:.2f}/10",
                f"{stats['avg_total_score']:.2f}/30",
                f"{stats['avg_total_score']/30*100:.1f}%"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        csv_file = f"{self.output_dir}/summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ 评估摘要: {csv_file}")
    
    def run_evaluation(self):
        """运行完整评估流程"""
        print(f"\n{'='*70}")
        print(f"开始完整评估流程")
        print(f"{'='*70}")
        
        # 1. 加载并合并模型
        model = self.load_and_merge_model()
        
        # 2. 评估模型
        results = self.evaluate_model(model)
        
        # 3. 计算统计
        stats = self.calculate_statistics(results)
        
        # 4. 打印统计
        self.print_statistics(stats)
        
        # 5. 打印详细案例
        self.print_detailed_cases(results, num_low=3, num_high=3)
        
        # 6. 保存结果
        self.save_results(results, stats)
        
        # 7. 清理内存
        del model
        torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print(f"评估完成！")
        print(f"{'='*70}")
        print(f"结果保存在: {self.output_dir}/")
        
        return results, stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='集中式训练模型GPT评分评估')
    parser.add_argument('--base_model', type=str, default="unsloth/Qwen3-8B-Base",
                        help='基座模型名称')
    parser.add_argument('--lora_path', type=str, 
                        default="./qwen3_java_evaluator_lora_unsloth_low_lr/final_model",
                        
                        help='LoRA适配器路径')
    parser.add_argument('--test_data', type=str, 
                        default="./data/test_data.json",
                        help='测试数据路径')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API Key（可选，默认从环境变量读取）')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o-mini',
                        help='GPT模型名称')
    parser.add_argument('--output_dir', type=str, 
                        default='./centralized_evaluation_results_unsloth_low_lr',
                        help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CentralizedModelEvaluator(
        base_model_name=args.base_model,
        lora_path=args.lora_path,
        test_data_path=args.test_data,
        openai_api_key=args.api_key,
        gpt_model=args.gpt_model,
        output_dir=args.output_dir
    )
    
    # 运行评估
    results, stats = evaluator.run_evaluation()
    
    print(f"\n🎉 完成！查看结果:")
    print(f"  - 详细结果: {args.output_dir}/detailed_results.json")
    print(f"  - 统计数据: {args.output_dir}/statistics.json")
    print(f"  - 评估摘要: {args.output_dir}/summary.csv")


if __name__ == "__main__":
    main()