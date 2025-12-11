"""
联邦学习框架 - 多GPU并行训练版本
版本: v2.3 - 支持多GPU并行训练多个客户端
"""

#!/usr/bin/env python3
"""
多GPU并行训练诊断脚本
用于检测和诊断多GPU量化模型训练的问题
"""

"""
联邦学习框架 - 多GPU真正并行训练版本
版本: v2.4 - 使用多进程实现真正的并行训练
"""
"""
联邦学习框架 - 多GPU真正并行训练版本
版本: v2.4 - 使用多进程实现真正的并行训练
"""

import os
import torch
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset
from typing import List, Dict, Tuple
import numpy as np
import multiprocessing as mp


# ==================== 配置 ====================
class FederatedConfig:
    """联邦学习配置"""
    
    # 模型配置
    MODEL_NAME = "Qwen/Qwen3-4B-Base"
    
    # LoRA配置
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # 训练配置
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1e-4
    MAX_SEQ_LENGTH = 1536
    
    # 联邦学习配置
    NUM_CLIENTS = 2
    NUM_ROUNDS = 1
    
    # 多GPU配置
    NUM_GPUS = torch.cuda.device_count()
    PARALLEL_CLIENTS = True  # ⭐ 真正的并行训练
    
    # 各算法特有参数
    FEDPROX_MU = 0.01
    FEDADAM_BETA1 = 0.9
    FEDADAM_BETA2 = 0.99
    FEDADAM_TAU = 1e-3
    
    # 量化配置
    USE_4BIT = True
    USE_NESTED_QUANT = True
    USE_GRADIENT_CHECKPOINTING = True
    
    # 输出目录
    OUTPUT_DIR = "./java_error_federated_results"
    DATA_DIR = "./data"


def check_gpu_capability():
    """检查GPU能力"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU才能运行此脚本")
    
    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 块GPU:")
    
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        compute_capability = torch.cuda.get_device_capability(i)
        memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {device_name}")
        print(f"    计算能力: {compute_capability[0]}.{compute_capability[1]}")
        print(f"    显存: {memory:.1f} GB")
    
    compute_capability = torch.cuda.get_device_capability(0)
    supports_bf16 = compute_capability[0] >= 8
    
    if supports_bf16:
        print(f"\n  ✅ GPU支持bfloat16")
        compute_dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
    else:
        print(f"\n  ⚠️ GPU不支持bfloat16，使用float16")
        compute_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True
    
    return compute_dtype, use_bf16, use_fp16, num_gpus


# ==================== 数据加载 ====================
def preprocess_function(examples, tokenizer, max_length=1536):
    """预处理函数"""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def load_client_dataset(client_id: int, data_dir: str, tokenizer, max_length: int):
    """加载单个客户端的数据（用于多进程）"""
    data_file = f"{data_dir}/client_{client_id}.json"
    
    dataset = load_dataset('json', data_files=data_file, split='train')
    
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing client {client_id}"
    )
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return dataset


# ==================== 参数操作 ====================
def get_model_parameters(model):
    """获取LoRA参数"""
    parameters = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters[name] = param.cpu().detach().clone()
    return parameters


def set_model_parameters(model, parameters):
    """设置LoRA参数"""
    for name, param in model.named_parameters():
        if name in parameters:
            param.data = parameters[name].to(param.device)


# ==================== FedAvg ====================
def aggregate_fedavg(client_params_list: List[Dict], 
                     client_sizes: List[int]) -> Dict:
    """FedAvg聚合"""
    total_size = sum(client_sizes)
    aggregated_params = {}
    
    for name in client_params_list[0].keys():
        aggregated_params[name] = torch.zeros_like(client_params_list[0][name])
        for i, client_params in enumerate(client_params_list):
            weight = client_sizes[i] / total_size
            aggregated_params[name] += client_params[name] * weight
    
    return aggregated_params


# ==================== FedProx ====================
class FedProxTrainer(Trainer):
    """FedProx Trainer"""
    def __init__(self, *args, mu=0.01, global_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.global_params = global_params
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(**inputs)
        loss = outputs.loss
        
        if self.global_params is not None:
            proximal_term = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.global_params:
                    proximal_term += ((param - self.global_params[name].to(param.device)) ** 2).sum()
            
            loss = loss + (self.mu / 2) * proximal_term
        
        return (loss, outputs) if return_outputs else loss



# ==================== 并行客户端训练（关键！）====================
def train_client_process(
    client_id: int,
    device_id: int,
    global_params_dict: Dict,
    config_dict: Dict,
    precision_info: Dict,
    method: str = "fedavg"
) -> Tuple[int, Dict, float, int]:
    """
    在独立进程中训练客户端
    ⭐ 这是实现并行训练的关键函数
    """
    
    # 导入必要的库（在子进程中）
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
        Trainer,
        default_data_collator,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    
    # ⭐ 关键：在任何CUDA操作之前设置设备
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # 现在这个进程只能看到一个GPU（编号为0）
    torch.cuda.set_device(0)
    
    print(f"\n[进程 {os.getpid()}] 客户端 {client_id} 在物理GPU {device_id}（进程内GPU 0）上训练")
    
    # 重建config对象
    config = FederatedConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # 确定精度
    if precision_info['use_bf16']:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    # ⭐ 在这个进程中，GPU 0就是我们想要的GPU
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': 0},  # 进程内的GPU 0
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        use_cache=False,
    )
    
    # 准备训练
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING
    )
    
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    # LoRA
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # 加载数据
    dataset = load_client_dataset(client_id, config.DATA_DIR, tokenizer, config.MAX_SEQ_LENGTH)
    dataset_size = len(dataset)
    
    # 加载全局参数
    set_model_parameters(model, global_params_dict)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR}/{method}/client_{client_id}",
        num_train_epochs=config.LOCAL_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=precision_info['use_fp16'],
        bf16=precision_info['use_bf16'],
        optim="paged_adamw_8bit",
        gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    data_collator = default_data_collator
    
    if method == "fedprox":
        trainer = FedProxTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            mu=config.FEDPROX_MU,
            global_params=global_params_dict,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
    
    # 训练
    result = trainer.train()
    train_loss = result.training_loss
    
    # 获取训练后的参数
    client_params = get_model_parameters(model)
    
    print(f"✓ 客户端 {client_id} 完成 (物理GPU {device_id}) - Loss: {train_loss:.4f}")
    
    # 清理
    del model
    del trainer
    del dataset
    torch.cuda.empty_cache()
    
    return client_id, client_params, train_loss, dataset_size


# ==================== 训练循环（并行版）====================
def train_federated_parallel(
    base_model,
    tokenizer,
    client_datasets,
    config: FederatedConfig,
    precision_info: dict,
    method: str = "fedavg"
):
    """
    联邦学习训练循环 - 真正的并行版本
    ⭐ 使用多进程池同时训练多个客户端
    """
    
    print(f"\n{'='*70}")
    print(f"开始 {method.upper()} 训练（多进程并行）")
    print(f"{'='*70}")
    print(f"GPU数量: {config.NUM_GPUS}")
    print(f"客户端数量: {config.NUM_CLIENTS}")
    print(f"⭐ 并行训练: 是")
    
    optimizer = None
    if method == "fedadam":
        optimizer = FedAdamOptimizer(
            beta1=config.FEDADAM_BETA1,
            beta2=config.FEDADAM_BETA2,
            tau=config.FEDADAM_TAU
        )
    
    training_history = {
        "method": method,
        "rounds": [],
        "avg_losses": [],
        "client_losses": [],
        "round_times": []
    }
    
    # 准备config字典（用于传递给子进程）
    config_dict = {
        'MODEL_NAME': config.MODEL_NAME,
        'LORA_R': config.LORA_R,
        'LORA_ALPHA': config.LORA_ALPHA,
        'LORA_DROPOUT': config.LORA_DROPOUT,
        'LORA_TARGET_MODULES': config.LORA_TARGET_MODULES,
        'LOCAL_EPOCHS': config.LOCAL_EPOCHS,
        'BATCH_SIZE': config.BATCH_SIZE,
        'GRADIENT_ACCUMULATION_STEPS': config.GRADIENT_ACCUMULATION_STEPS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'MAX_SEQ_LENGTH': config.MAX_SEQ_LENGTH,
        'USE_NESTED_QUANT': config.USE_NESTED_QUANT,
        'USE_GRADIENT_CHECKPOINTING': config.USE_GRADIENT_CHECKPOINTING,
        'OUTPUT_DIR': config.OUTPUT_DIR,
        'DATA_DIR': config.DATA_DIR,
        'FEDPROX_MU': config.FEDPROX_MU,
    }
    
    for round_idx in range(config.NUM_ROUNDS):
        round_start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"# 轮次 {round_idx + 1}/{config.NUM_ROUNDS} - {method.upper()}")
        print(f"{'#'*70}")
        
        # 获取全局参数
        global_params = get_model_parameters(base_model)
        
        # ⭐ 使用进程池并行训练
        print(f"\n并行训练 {config.NUM_CLIENTS} 个客户端...")
        
        # 准备任务列表
        tasks = []
        for client_id in range(config.NUM_CLIENTS):
            device_id = client_id % config.NUM_GPUS
            tasks.append((client_id, device_id, global_params, config_dict, precision_info, method))
        
        # 创建进程池并并行执行
        # ⭐ 使用上下文管理器确保资源正确清理
        try:
            with mp.Pool(processes=min(config.NUM_GPUS, config.NUM_CLIENTS)) as pool:
                results = pool.starmap(train_client_process, tasks)
                pool.close()
                pool.join()
        except Exception as e:
            print(f"并行训练出错: {e}")
            raise
        
        # 收集结果
        client_params_list = [None] * config.NUM_CLIENTS
        client_sizes = [0] * config.NUM_CLIENTS
        round_losses = [0.0] * config.NUM_CLIENTS
        
        for client_id, client_params, loss, size in results:
            client_params_list[client_id] = client_params
            client_sizes[client_id] = size
            round_losses[client_id] = loss
        
        # 聚合
        print(f"\n聚合 {config.NUM_CLIENTS} 个客户端...")
        
        if method == "fedavg" or method == "fedprox":
            aggregated_params = aggregate_fedavg(client_params_list, client_sizes)
        elif method == "fedadam":
            aggregated_params = optimizer.step(global_params, client_params_list, client_sizes)
        
        set_model_parameters(base_model, aggregated_params)
        
        # 统计
        avg_loss = sum(round_losses) / len(round_losses)
        round_time = time.time() - round_start_time
        
        print(f"\n轮次 {round_idx + 1} 汇总:")
        print(f"  平均Loss: {avg_loss:.4f}")
        print(f"  Loss范围: [{min(round_losses):.4f}, {max(round_losses):.4f}]")
        print(f"  耗时: {round_time:.1f}秒")
        
        training_history["rounds"].append(round_idx + 1)
        training_history["avg_losses"].append(avg_loss)
        training_history["client_losses"].append(round_losses)
        training_history["round_times"].append(round_time)
        
        os.makedirs(f"{config.OUTPUT_DIR}/{method}", exist_ok=True)
        with open(f"{config.OUTPUT_DIR}/{method}/training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
    
    return base_model, training_history


# ==================== 主函数 ====================
def main():
    """主函数"""

    
    config = FederatedConfig()
    
    # 检查GPU
    compute_dtype, use_bf16, use_fp16, num_gpus = check_gpu_capability()
    config.NUM_GPUS = num_gpus
    
    if num_gpus > 1 and config.PARALLEL_CLIENTS:
        print(f"\n⭐⭐⭐ 将使用 {num_gpus} 块GPU并行训练客户端！")
        print(f"预计训练时间将减少 ~{min(num_gpus, config.NUM_CLIENTS) * 100 // config.NUM_CLIENTS}%")
    else:
        print(f"\n使用单GPU顺序训练")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据（只是为了获取大小，实际训练时在子进程中加载）
    print("\n加载客户端数据...")
    client_datasets = []
    for i in range(config.NUM_CLIENTS):
        dataset = load_client_dataset(i, config.DATA_DIR, tokenizer, config.MAX_SEQ_LENGTH)
        client_datasets.append(dataset)
        print(f"✓ 客户端 {i}: {len(dataset)} 样本")
    
    # 初始化基础模型（用于参数聚合）
    print("\n初始化基础模型...")
    torch.cuda.set_device(0)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        use_cache=False,
    )
    
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING
    )
    
    if config.USE_GRADIENT_CHECKPOINTING:
        base_model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    base_model = get_peft_model(base_model, peft_config)
    base_model.config.pad_token_id = tokenizer.eos_token_id
    
    print("✅ 基础模型初始化完成")
    
    precision_info = {
        'use_bf16': use_bf16,
        'use_fp16': use_fp16,
        'compute_dtype': str(compute_dtype)
    }
    
    # 保存配置
    config_dict = {
        "model": config.MODEL_NAME,
        "num_gpus": config.NUM_GPUS,
        "parallel_training": config.PARALLEL_CLIENTS,
        "num_clients": config.NUM_CLIENTS,
        "num_rounds": config.NUM_ROUNDS,
        "methods": ["fedavg", "fedprox", "fedadam"],
    }
    with open(f"{config.OUTPUT_DIR}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    all_histories = {}
    
    # 训练所有算法
    for method in ["fedavg", "fedprox"]:
        print("\n" + "="*70)
        print(f"实验: {method.upper()}")
        print("="*70)
        
        # 重新初始化模型
        print(f"为{method}初始化新模型...")
        method_model, _, _ = (base_model, tokenizer, precision_info)  # 使用base_model
        
        method_model, history = train_federated_parallel(
            method_model,
            tokenizer,
            client_datasets,
            config,
            precision_info,
            method=method
        )
        all_histories[method] = history
        
        print(f"\n保存{method.upper()}模型...")
        method_model.save_pretrained(f"{config.OUTPUT_DIR}/{method}/final_model")
        tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/{method}/final_model")
        
        torch.cuda.empty_cache()
        time.sleep(2)
    
    # 结果对比
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)
    
    comparison = {}
    print("\n最终Loss:")
    for method, history in all_histories.items():
        final_loss = history['avg_losses'][-1]
        avg_time = np.mean(history['round_times'])
        print(f"  {method.upper():10s}: {final_loss:.4f} (平均时间: {avg_time:.1f}秒/轮)")
        
        comparison[method] = {
            "final_loss": final_loss,
            "avg_round_time": avg_time,
            "total_time": sum(history['round_times'])
        }
    
    with open(f"{config.OUTPUT_DIR}/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":

    # 设置multiprocessing启动方式为spawn
    mp.set_start_method('spawn', force=True)
    main()