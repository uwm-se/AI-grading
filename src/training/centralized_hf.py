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
import numpy as np


# ==================== 配置 ====================
class CentralizedConfig:
    """集中式训练配置（与联邦学习超参数完全一致）"""
    
    # 模型配置
    MODEL_NAME = "Qwen/Qwen3-8B-Base"
    
    # LoRA配置（与联邦学习完全一致）
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # 训练配置（与联邦学习完全一致）
    NUM_EPOCHS = 3  # 对应联邦学习的 LOCAL_EPOCHS * NUM_ROUNDS = 3 * 1
    BATCH_SIZE = 1  # 与联邦学习单个client一致
    GRADIENT_ACCUMULATION_STEPS = 8  # 与联邦学习完全一致
    LEARNING_RATE = 1e-4  # 与联邦学习完全一致
    MAX_SEQ_LENGTH = 1536  # 与联邦学习完全一致
    
    # 量化配置（与联邦学习完全一致）
    USE_4BIT = True
    USE_NESTED_QUANT = True
    USE_GRADIENT_CHECKPOINTING = True
    
    # 数据配置
    DATA_DIR = "./data"
    
    # 输出目录
    OUTPUT_DIR = "./java_error_centralized_results_8b"


def check_gpu_capability():
    """检查GPU能力"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU才能运行此脚本")
    
    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\n检测到GPU:")
    print(f"  设备: {device_name}")
    print(f"  计算能力: {compute_capability[0]}.{compute_capability[1]}")
    print(f"  显存: {memory:.1f} GB")
    
    # 判断是否支持bfloat16
    supports_bf16 = compute_capability[0] >= 8
    
    if supports_bf16:
        print(f"  ✅ GPU支持bfloat16")
        compute_dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
    else:
        print(f"  ⚠️ GPU不支持bfloat16，使用float16")
        compute_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True
    
    return compute_dtype, use_bf16, use_fp16


# ==================== 数据加载 ====================
def preprocess_function(examples, tokenizer, max_length=1536):
    """预处理函数（与联邦学习完全一致）"""
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


def load_train_data(data_dir: str, tokenizer, max_length: int):
    """
    加载训练数据
    直接使用完整的训练集（联邦学习中被分割为多个client）
    """
    print(f"\n加载训练数据...")
    
    train_file = f"{data_dir}/new_train_data_message.json"
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"找不到训练数据文件: {train_file}")
    
    dataset = load_dataset('json', data_files=train_file, split='train')
    
    print(f"  原始样本数: {len(dataset)}")
    
    # 预处理
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing training data"
    )
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print(f"\n✅ 训练数据加载完成:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  注: 联邦学习中此数据被分割为多个客户端")
    
    return dataset


def load_validation_data(data_dir: str, tokenizer, max_length: int):
    """
    加载验证集数据
    """
    val_file = f"{data_dir}/valid.json"
    
    if not os.path.exists(val_file):
        print(f"\n⚠️  警告: 找不到验证集文件 {val_file}")
        return None
    
    print(f"\n加载验证集...")
    
    dataset = load_dataset('json', data_files=val_file, split='train')
    
    # 预处理
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing validation set"
    )
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print(f"✅ 验证集加载完成:")
    print(f"  样本数: {len(dataset)}")
    
    return dataset


# ==================== 训练函数 ====================
def train_centralized(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: CentralizedConfig,
    precision_info: dict
):
    """
    集中式训练（带验证集评估）
    超参数与联邦学习完全一致
    """
    
    print(f"\n{'='*70}")
    print(f"开始集中式训练（Centralized Training with Validation）")
    print(f"{'='*70}")
    
    # 设置评估策略
    if eval_dataset is not None:
        evaluation_strategy = "steps"  # 每N步评估一次
        eval_steps = 50  # 可以根据数据量调整
        load_best_model_at_end = True
        metric_for_best_model = "eval_loss"
        greater_is_better = False
        print(f"✅ 启用验证集评估 (每 {eval_steps} 步)")
    else:
        evaluation_strategy = "no"
        eval_steps = None
        load_best_model_at_end = False
        metric_for_best_model = None
        greater_is_better = None
        print(f"⚠️  未找到验证集，不进行评估")
    
    # 训练参数（与联邦学习完全一致）
    training_args = TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR}/checkpoints",
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,  # 验证时也用相同batch size
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=precision_info['use_fp16'],
        bf16=precision_info['use_bf16'],
        optim="paged_adamw_8bit",
        gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        logging_steps=10,
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        save_strategy="steps" if eval_dataset is not None else "epoch",
        save_steps=eval_steps if eval_dataset is not None else None,
        save_total_limit=3,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # 数据整理器
    data_collator = default_data_collator
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 添加验证集
        data_collator=data_collator,
    )
    
    # 训练前信息
    print(f"\n训练配置:")
    print(f"  训练样本数: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  验证样本数: {len(eval_dataset)}")
    print(f"  训练轮次: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  梯度累积步数: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  实际Batch Size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  最大序列长度: {config.MAX_SEQ_LENGTH}")
    
    # 训练
    print(f"\n开始训练...")
    start_time = time.time()
    result = trainer.train()
    end_time = time.time()
    
    training_time = end_time - start_time
    
    # 最终评估
    final_metrics = {}
    if eval_dataset is not None:
        print(f"\n进行最终评估...")
        eval_results = trainer.evaluate()
        final_metrics = eval_results
        print(f"✅ 最终验证Loss: {eval_results['eval_loss']:.4f}")
    
    # 训练历史
    training_history = {
        "method": "centralized",
        "model": config.MODEL_NAME,
        "total_train_samples": len(train_dataset),
        "total_eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "final_train_loss": result.training_loss,
        "final_eval_metrics": final_metrics,
        "total_training_time": training_time,
        "log_history": trainer.state.log_history
    }
    
    print(f"\n{'='*70}")
    print(f"训练完成!")
    print(f"{'='*70}")
    print(f"  最终训练Loss: {result.training_loss:.4f}")
    if eval_dataset is not None:
        print(f"  最终验证Loss: {final_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  总训练时间: {training_time:.1f}秒 ({training_time/60:.1f}分钟)")
    print(f"  平均每epoch: {training_time/config.NUM_EPOCHS:.1f}秒")
    
    return model, training_history


# ==================== 主函数 ====================
def main():
    """主函数"""
    
    print("="*70)
    print("集中式训练 (Centralized Training)")
    print("作为联邦学习的Baseline对比")
    print("="*70)
    print("⚠️  注意: 所有超参数与联邦学习完全一致")
    print("="*70)
    
    config = CentralizedConfig()
    
    # 检查GPU
    compute_dtype, use_bf16, use_fp16 = check_gpu_capability()
    
    precision_info = {
        'use_bf16': use_bf16,
        'use_fp16': use_fp16,
        'compute_dtype': str(compute_dtype)
    }
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 初始化tokenizer
    print("\n初始化Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Tokenizer初始化完成")
    
    # 加载训练数据
    train_dataset = load_train_data(
        config.DATA_DIR,
        #"./data/new_train_data_message.json",
        tokenizer,
        config.MAX_SEQ_LENGTH
    )
    
    # 加载验证集
    eval_dataset = load_validation_data(
        config.DATA_DIR,
        #"./data/valid.json",
        tokenizer,
        config.MAX_SEQ_LENGTH
    )
    
    # 初始化模型
    print("\n初始化模型...")
    torch.cuda.set_device(0)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=config.USE_NESTED_QUANT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': 0},
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
    
    # 添加LoRA
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n✅ 模型初始化完成")
    print(f"  基模型: {config.MODEL_NAME}")
    print(f"  LoRA Rank: {config.LORA_R}")
    print(f"  LoRA Alpha: {config.LORA_ALPHA}")
    print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  总参数: {total_params:,}")
    
    # 保存配置
    config_dict = {
        "method": "centralized",
        "model": config.MODEL_NAME,
        "lora_config": {
            "r": config.LORA_R,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "target_modules": config.LORA_TARGET_MODULES
        },
        "training_config": {
            "num_epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": config.LEARNING_RATE,
            "max_seq_length": config.MAX_SEQ_LENGTH,
        },
        "data_config": {
            "train_file": "new_train_data_message.json",
            "validation_file": "valid.json",
            "total_train_samples": len(train_dataset),
            "total_eval_samples": len(eval_dataset) if eval_dataset is not None else 0
        },
        "note": "超参数与联邦学习完全一致，用于公平对比"
    }
    
    with open(f"{config.OUTPUT_DIR}/config.json", "w", encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # 训练
    model, training_history = train_centralized(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,  # 添加验证集
        config,
        precision_info
    )
    
    # 保存模型
    print(f"\n保存模型...")
    model.save_pretrained(f"{config.OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/final_model")
    print(f"✓ 模型已保存到: {config.OUTPUT_DIR}/final_model")
    
    # 保存训练历史
    with open(f"{config.OUTPUT_DIR}/training_history.json", "w", encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # 结果总结
    print("\n" + "="*70)
    print("训练结果总结")
    print("="*70)
    print(f"训练方法: 集中式训练 (Centralized)")
    print(f"模型: {config.MODEL_NAME}")
    print(f"数据: 完整训练集 (new_train_data.json)")
    print(f"训练样本数: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"验证样本数: {len(eval_dataset)}")
    print(f"训练轮次: {config.NUM_EPOCHS}")
    print(f"最终训练Loss: {training_history['final_train_loss']:.4f}")
    if eval_dataset is not None and training_history['final_eval_metrics']:
        print(f"最终验证Loss: {training_history['final_eval_metrics']['eval_loss']:.4f}")
    print(f"总训练时间: {training_history['total_training_time']:.1f}秒")
    print(f"\n结果保存在: {config.OUTPUT_DIR}/")
    print("="*70)
    
    print("\n💡 提示:")
    print("  - 此集中式训练使用与联邦学习完全相同的超参数")
    print("  - 使用完整的训练集（联邦学习中被分割为多个客户端）")
    print("  - 可以与联邦学习结果进行公平对比")
    if eval_dataset is not None:
        print("  - 已使用验证集监控训练质量，防止过拟合")
        print("  - 自动保存验证Loss最低的模型")
    print("  - 对比指标: 训练Loss、验证Loss、训练时间、收敛速度等")


if __name__ == "__main__":
    main()