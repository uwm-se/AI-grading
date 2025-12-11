import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments,AutoTokenizer, AutoModelForCausalLM


# ==================== 配置参数 ====================
# 模型配置（与联邦学习完全一致）
MODEL_NAME = "unsloth/Qwen3-8B-Base"  # 使用 Qwen3-8B-Base
MAX_SEQ_LENGTH = 1536  # 与联邦学习完全一致

# 量化配置（与联邦学习完全一致）
LOAD_IN_4BIT = True
USE_NESTED_QUANT = True  # 使用嵌套量化
USE_GRADIENT_CHECKPOINTING = True

# LoRA 参数（与联邦学习完全一致）
LORA_R = 32 #16  # LoRA rank
LORA_ALPHA = 64 #32  # LoRA alpha
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 训练参数（与联邦学习完全一致）
OUTPUT_DIR = "./qwen3_java_evaluator_lora_unsloth_low_lr"
NUM_TRAIN_EPOCHS = 5  # 对应联邦学习的 LOCAL_EPOCHS * NUM_ROUNDS = 3 * 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 与联邦学习单个client一致
PER_DEVICE_EVAL_BATCH_SIZE = 1  # 与训练batch size保持一致
GRADIENT_ACCUMULATION_STEPS = 8  # 与联邦学习完全一致（有效 batch size = 1 * 8 = 8）
LEARNING_RATE = 5e-5  # 与联邦学习完全一致
WARMUP_STEPS = 100
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 100

# ==================== 加载数据 ====================
def load_json_data(file_path):
    """加载 JSONL 格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def format_chat_template(sample):
    """将数据格式化为 Qwen 的 chat 格式"""
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["user_prompt"]},
        {"role": "assistant", "content": sample["feedback"]}
    ]
    return {"messages": messages}

def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        texts.append(text)
    return {"text": texts}

print("📚 加载数据...")
train_data = load_json_data("./data/new_train_data.json")
valid_data = load_json_data("./data/valid_data.json")

print(f"训练数据: {len(train_data)} 条")
print(f"验证数据: {len(valid_data)} 条")

# 转换为 Dataset
train_dataset = Dataset.from_list([format_chat_template(d) for d in train_data])
valid_dataset = Dataset.from_list([format_chat_template(d) for d in valid_data])
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True)

print(train_dataset[:2])

# ==================== 加载模型 ====================
print(f"\n🤖 加载模型: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # 自动选择
    load_in_4bit=LOAD_IN_4BIT,
)


# 配置 LoRA
print("🔧 配置 LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth" if USE_GRADIENT_CHECKPOINTING else False,
    random_state=42,
)

# 打印可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

# ==================== 训练配置 ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",  # 8-bit AdamW 节省显存
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    report_to="none",  # 不使用 wandb
)

# ==================== 创建 Trainer ====================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    #dataset_text_field="messages",  # 使用 messages 字段
    packing=False,  # 不使用 packing，保持对话完整性
)

# ==================== 开始训练 ====================
print("\n🚀 开始训练...")
trainer.train()

# ==================== 保存模型 ====================
print("\n💾 保存模型...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")


print("\n✅ 训练完成！")
print(f"模型保存在: {OUTPUT_DIR}/final_model")
