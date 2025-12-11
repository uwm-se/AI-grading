"""
联邦学习框架 - 基于Flower库的FedAdam实现
版本: v3.0
"""

import os
import torch
import json
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

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
    
    # 验证配置
    EVAL_EVERY_N_ROUNDS = 1  # 每N轮验证一次
    EARLY_STOPPING_PATIENCE = 3  # 验证loss连续N轮不下降则停止
    EARLY_STOPPING_ENABLED = True  # 是否启用early stopping
    
    # FedAdam参数
    FEDADAM_BETA1 = 0.9
    FEDADAM_BETA2 = 0.99
    FEDADAM_TAU = 1e-3
    
    # 量化配置
    USE_4BIT = True
    USE_NESTED_QUANT = True
    USE_GRADIENT_CHECKPOINTING = True
    
    # 输出目录
    OUTPUT_DIR = "./java_error_federated_results/fedadam_flower"
    DATA_DIR = "./data"


CONFIG = FederatedConfig()


# ==================== 工具函数 ====================
def check_gpu_capability():
    """检查GPU能力"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要GPU才能运行此脚本")
    
    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 块GPU:")
    
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {device_name} ({memory:.1f} GB)")
    
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
    """加载单个客户端的数据"""
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


def load_eval_dataset(data_dir: str, tokenizer, max_length: int):
    """加载验证集"""
    eval_file = f"{data_dir}/valid.json"
    
    if not os.path.exists(eval_file):
        print(f"  ⚠️ 验证集文件不存在: {eval_file}")
        return None
    
    dataset = load_dataset('json', data_files=eval_file, split='train')
    
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing eval dataset"
    )
    
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return dataset


def get_lora_parameters(model) -> List[np.ndarray]:
    """获取LoRA参数作为numpy数组列表"""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param.cpu().detach().numpy())
    return params


def get_lora_param_names(model) -> List[str]:
    """获取LoRA参数名称"""
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
    return names


def set_lora_parameters(model, parameters: List[np.ndarray]):
    """设置LoRA参数"""
    param_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = torch.tensor(parameters[param_idx]).to(param.device)
            param_idx += 1


def evaluate_model(model, eval_dataset, config: FederatedConfig, precision_info: dict):
    """
    评估模型性能
    返回验证loss和perplexity
    """
    if eval_dataset is None:
        return None, None
    
    model.eval()
    
    eval_args = TrainingArguments(
        output_dir=f"{config.OUTPUT_DIR}/eval_tmp",
        per_device_eval_batch_size=config.BATCH_SIZE,
        fp16=precision_info['use_fp16'],
        bf16=precision_info['use_bf16'],
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    
    with torch.no_grad():
        metrics = trainer.evaluate()
    
    eval_loss = metrics.get("eval_loss", float('inf'))
    perplexity = np.exp(eval_loss) if eval_loss < 100 else float('inf')
    
    model.train()
    
    return eval_loss, perplexity


# ==================== FedAdam Strategy ====================
class FedAdamStrategy(Strategy):
    """
    FedAdam聚合策略
    实现了服务端的自适应优化
    """
    
    def __init__(
        self,
        initial_parameters: Parameters,
        param_names: List[str],
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
    ):
        self.initial_parameters = initial_parameters
        self.param_names = param_names
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        
        # 初始化动量
        self.m: Optional[List[np.ndarray]] = None
        self.v: Optional[List[np.ndarray]] = None
        self.t = 0
        
        # 当前全局参数
        self.current_parameters = initial_parameters
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """初始化全局参数"""
        return self.initial_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """配置客户端训练"""
        
        # 采样客户端
        sample_size = self.min_fit_clients
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients,
        )
        
        # 创建FitIns
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        FedAdam聚合
        """
        if not results:
            return None, {}
        
        self.t += 1
        
        # 获取当前全局参数
        global_params = parameters_to_ndarrays(self.current_parameters)
        
        # 收集客户端参数和样本数
        client_params_list = []
        client_sizes = []
        
        for client_proxy, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            client_params_list.append(client_params)
            client_sizes.append(fit_res.num_examples)
        
        total_size = sum(client_sizes)
        
        # 计算伪梯度 (global - weighted_avg_client)
        pseudo_gradient = []
        for i in range(len(global_params)):
            grad = np.zeros_like(global_params[i])
            for j, client_params in enumerate(client_params_list):
                weight = client_sizes[j] / total_size
                grad += weight * (global_params[i] - client_params[i])
            pseudo_gradient.append(grad)
        
        # 初始化动量（如果需要）
        if self.m is None:
            self.m = [np.zeros_like(p) for p in global_params]
            self.v = [np.zeros_like(p) for p in global_params]
        
        # FedAdam更新
        new_params = []
        for i in range(len(global_params)):
            # 更新一阶动量
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * pseudo_gradient[i]
            # 更新二阶动量
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (pseudo_gradient[i] ** 2)
            
            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            new_param = global_params[i] - self.tau * m_hat / (np.sqrt(v_hat) + 1e-8)
            new_params.append(new_param)
        
        # 更新当前参数
        self.current_parameters = ndarrays_to_parameters(new_params)
        
        # 计算平均loss
        avg_loss = sum(fit_res.metrics.get("train_loss", 0) for _, fit_res in results) / len(results)
        
        print(f"\n  轮次 {server_round} FedAdam聚合完成")
        print(f"  平均Loss: {avg_loss:.4f}")
        
        return self.current_parameters, {"avg_loss": avg_loss}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """配置评估（本实验不使用）"""
        return []
    
    def aggregate_evaluate(self, server_round, results, failures):
        """聚合评估结果（本实验不使用）"""
        return None, {}
    
    def evaluate(self, server_round, parameters):
        """服务端评估（本实验不使用）"""
        return None


# ==================== Flower Client ====================
class FlowerClient(fl.client.NumPyClient):
    """
    Flower客户端
    负责本地训练
    """
    
    def __init__(
        self,
        client_id: int,
        model,
        tokenizer,
        dataset,
        config: FederatedConfig,
        precision_info: dict,
    ):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.precision_info = precision_info
        self.device = next(model.parameters()).device
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """获取模型参数"""
        return get_lora_parameters(self.model)
    
    def set_parameters(self, parameters: NDArrays):
        """设置模型参数"""
        set_lora_parameters(self.model, parameters)
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """本地训练"""
        
        server_round = config.get("server_round", 0)
        print(f"\n  客户端 {self.client_id} 开始训练 (轮次 {server_round})")
        
        # 设置全局参数
        self.set_parameters(parameters)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/client_{self.client_id}",
            num_train_epochs=self.config.LOCAL_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=self.precision_info['use_fp16'],
            bf16=self.precision_info['use_bf16'],
            optim="paged_adamw_8bit",
            gradient_checkpointing=self.config.USE_GRADIENT_CHECKPOINTING,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=default_data_collator,
        )
        
        # 训练
        result = trainer.train()
        train_loss = result.training_loss
        
        print(f"  ✓ 客户端 {self.client_id} 完成 - Loss: {train_loss:.4f}")
        
        # 返回更新后的参数
        return (
            self.get_parameters(config={}),
            len(self.dataset),
            {"train_loss": train_loss},
        )
    
    def evaluate(self, parameters, config):
        """评估（本实验不使用）"""
        return 0.0, 0, {}


# ==================== 客户端工厂 ====================
def create_client_fn(
    client_id: int,
    model,
    tokenizer,
    dataset,
    config: FederatedConfig,
    precision_info: dict,
):
    """创建客户端实例"""
    return FlowerClient(
        client_id=client_id,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        precision_info=precision_info,
    )


# ==================== 模拟联邦学习 ====================
def run_simulation():
    """
    运行联邦学习模拟
    使用Flower的模拟模式在单机上模拟多客户端
    """
    
    print("="*70)
    print("联邦学习实验 - Flower FedAdam")
    print("版本: v3.0")
    print("="*70)
    
    # 检查GPU
    compute_dtype, use_bf16, use_fp16, num_gpus = check_gpu_capability()
    
    precision_info = {
        'use_bf16': use_bf16,
        'use_fp16': use_fp16,
    }
    
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    # 初始化tokenizer
    print("\n初始化Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    print("\n加载客户端数据...")
    client_datasets = []
    for i in range(CONFIG.NUM_CLIENTS):
        dataset = load_client_dataset(i, CONFIG.DATA_DIR, tokenizer, CONFIG.MAX_SEQ_LENGTH)
        client_datasets.append(dataset)
        print(f"  ✓ 客户端 {i}: {len(dataset)} 样本")
    
    # 加载验证集
    print("\n加载验证集...")
    eval_dataset = load_eval_dataset(CONFIG.DATA_DIR, tokenizer, CONFIG.MAX_SEQ_LENGTH)
    if eval_dataset is not None:
        print(f"  ✓ 验证集: {len(eval_dataset)} 样本")
    else:
        print("  ⚠️ 未找到验证集，将跳过验证步骤")
    
    # 初始化模型
    print("\n初始化模型...")
    torch.cuda.set_device(0)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=CONFIG.USE_NESTED_QUANT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        use_cache=False,
    )
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=CONFIG.USE_GRADIENT_CHECKPOINTING
    )
    
    if CONFIG.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        r=CONFIG.LORA_R,
        lora_alpha=CONFIG.LORA_ALPHA,
        lora_dropout=CONFIG.LORA_DROPOUT,
        target_modules=CONFIG.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    print("  ✅ 模型初始化完成")
    model.print_trainable_parameters()
    
    # 获取初始参数
    initial_params = get_lora_parameters(model)
    param_names = get_lora_param_names(model)
    
    print(f"\n可训练参数数量: {len(initial_params)}")
    
    # 创建FedAdam策略
    strategy = FedAdamStrategy(
        initial_parameters=ndarrays_to_parameters(initial_params),
        param_names=param_names,
        beta1=CONFIG.FEDADAM_BETA1,
        beta2=CONFIG.FEDADAM_BETA2,
        tau=CONFIG.FEDADAM_TAU,
        min_fit_clients=CONFIG.NUM_CLIENTS,
        min_available_clients=CONFIG.NUM_CLIENTS,
    )
    
    # 创建客户端
    print("\n创建客户端...")
    clients = {}
    for i in range(CONFIG.NUM_CLIENTS):
        client = create_client_fn(
            client_id=i,
            model=model,  # 注意：在模拟模式下共享模型
            tokenizer=tokenizer,
            dataset=client_datasets[i],
            config=CONFIG,
            precision_info=precision_info,
        )
        clients[str(i)] = client
        print(f"  ✓ 客户端 {i} 创建完成")
    
    # 手动运行联邦学习循环（模拟模式）
    print("\n" + "="*70)
    print("开始FedAdam联邦学习训练")
    print("="*70)
    
    training_history = {
        "method": "fedadam_flower",
        "rounds": [],
        "avg_losses": [],
        "eval_losses": [],
        "perplexities": [],
    }
    
    # 初始验证
    if eval_dataset is not None:
        print("\n初始模型验证...")
        init_eval_loss, init_ppl = evaluate_model(model, eval_dataset, CONFIG, precision_info)
        print(f"  初始验证Loss: {init_eval_loss:.4f}, Perplexity: {init_ppl:.2f}")
        training_history["initial_eval_loss"] = init_eval_loss
        training_history["initial_perplexity"] = init_ppl
    
    # Early Stopping状态
    best_eval_loss = float('inf')
    patience_counter = 0
    best_parameters = None
    
    current_parameters = initial_params
    
    for round_idx in range(CONFIG.NUM_ROUNDS):
        print(f"\n{'#'*70}")
        print(f"# 轮次 {round_idx + 1}/{CONFIG.NUM_ROUNDS}")
        print(f"{'#'*70}")
        
        # 训练每个客户端
        fit_results = []
        for client_id, client in clients.items():
            updated_params, num_examples, metrics = client.fit(
                parameters=current_parameters,
                config={"server_round": round_idx + 1},
            )
            
            # 创建FitRes
            fit_res = FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(updated_params),
                num_examples=num_examples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))
        
        # 聚合
        print(f"\n聚合 {len(fit_results)} 个客户端...")
        aggregated_params, metrics = strategy.aggregate_fit(
            server_round=round_idx + 1,
            results=fit_results,
            failures=[],
        )
        
        # 更新当前参数
        current_parameters = parameters_to_ndarrays(aggregated_params)
        
        # 验证
        eval_loss, perplexity = None, None
        if eval_dataset is not None and (round_idx + 1) % CONFIG.EVAL_EVERY_N_ROUNDS == 0:
            print(f"\n验证轮次 {round_idx + 1}...")
            set_lora_parameters(model, current_parameters)
            eval_loss, perplexity = evaluate_model(model, eval_dataset, CONFIG, precision_info)
            print(f"  验证Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # Early Stopping检查
            if CONFIG.EARLY_STOPPING_ENABLED:
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    best_parameters = [p.copy() for p in current_parameters]
                    print(f"  ✓ 新的最佳模型! (验证Loss: {best_eval_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  ⚠️ 验证Loss未改善 ({patience_counter}/{CONFIG.EARLY_STOPPING_PATIENCE})")
        
        # 记录历史
        training_history["rounds"].append(round_idx + 1)
        training_history["avg_losses"].append(metrics.get("avg_loss", 0))
        training_history["eval_losses"].append(eval_loss)
        training_history["perplexities"].append(perplexity)
        
        # Early Stopping
        if CONFIG.EARLY_STOPPING_ENABLED and patience_counter >= CONFIG.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early Stopping: 验证Loss连续{CONFIG.EARLY_STOPPING_PATIENCE}轮未改善")
            if best_parameters is not None:
                current_parameters = best_parameters
                print("  恢复到最佳模型参数")
            break
    
    # 保存最终模型
    print("\n保存最终模型...")
    
    # 如果有最佳参数，使用最佳参数
    if best_parameters is not None and CONFIG.EARLY_STOPPING_ENABLED:
        print("  使用最佳验证Loss时的参数")
        set_lora_parameters(model, best_parameters)
        training_history["best_eval_loss"] = best_eval_loss
    else:
        set_lora_parameters(model, current_parameters)
    
    model.save_pretrained(f"{CONFIG.OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{CONFIG.OUTPUT_DIR}/final_model")
    
    # 保存训练历史
    with open(f"{CONFIG.OUTPUT_DIR}/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"完成轮次: {len(training_history['rounds'])}/{CONFIG.NUM_ROUNDS}")
    print(f"最终训练Loss: {training_history['avg_losses'][-1]:.4f}")
    
    # 显示验证信息
    final_eval_losses = [l for l in training_history['eval_losses'] if l is not None]
    if final_eval_losses:
        print(f"最终验证Loss: {final_eval_losses[-1]:.4f}")
        final_ppls = [p for p in training_history['perplexities'] if p is not None]
        if final_ppls:
            print(f"最终Perplexity: {final_ppls[-1]:.2f}")
        
        # 显示最佳模型信息
        if "best_eval_loss" in training_history:
            print(f"最佳验证Loss: {training_history['best_eval_loss']:.4f}")
        
        # 显示改进情况
        if "initial_eval_loss" in training_history:
            init_loss = training_history["initial_eval_loss"]
            final_loss = training_history.get("best_eval_loss", final_eval_losses[-1])
            improvement = (init_loss - final_loss) / init_loss * 100
            print(f"验证Loss改进: {improvement:.1f}%")
    
    print(f"\n结果保存在: {CONFIG.OUTPUT_DIR}/")
    print("="*70)
    
    return model, training_history


# ==================== 分布式模式（可选）====================
def start_server():
    """
    启动Flower服务器（分布式模式）
    用于真实的多机分布式训练
    """
    
    print("启动Flower服务器...")
    
    # 初始化模型获取参数
    compute_dtype, use_bf16, use_fp16, _ = check_gpu_capability()
    
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=CONFIG.USE_NESTED_QUANT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        use_cache=False,
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=CONFIG.USE_GRADIENT_CHECKPOINTING)
    
    peft_config = LoraConfig(
        r=CONFIG.LORA_R,
        lora_alpha=CONFIG.LORA_ALPHA,
        lora_dropout=CONFIG.LORA_DROPOUT,
        target_modules=CONFIG.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    initial_params = get_lora_parameters(model)
    param_names = get_lora_param_names(model)
    
    strategy = FedAdamStrategy(
        initial_parameters=ndarrays_to_parameters(initial_params),
        param_names=param_names,
        beta1=CONFIG.FEDADAM_BETA1,
        beta2=CONFIG.FEDADAM_BETA2,
        tau=CONFIG.FEDADAM_TAU,
        min_fit_clients=CONFIG.NUM_CLIENTS,
        min_available_clients=CONFIG.NUM_CLIENTS,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=CONFIG.NUM_ROUNDS),
        strategy=strategy,
    )


def start_client(client_id: int):
    """
    启动Flower客户端（分布式模式）
    用于真实的多机分布式训练
    """
    
    print(f"启动客户端 {client_id}...")
    
    compute_dtype, use_bf16, use_fp16, _ = check_gpu_capability()
    
    precision_info = {
        'use_bf16': use_bf16,
        'use_fp16': use_fp16,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_client_dataset(client_id, CONFIG.DATA_DIR, tokenizer, CONFIG.MAX_SEQ_LENGTH)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=CONFIG.USE_NESTED_QUANT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        use_cache=False,
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=CONFIG.USE_GRADIENT_CHECKPOINTING)
    
    if CONFIG.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        r=CONFIG.LORA_R,
        lora_alpha=CONFIG.LORA_ALPHA,
        lora_dropout=CONFIG.LORA_DROPOUT,
        target_modules=CONFIG.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    client = FlowerClient(
        client_id=client_id,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=CONFIG,
        precision_info=precision_info,
    )
    
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
    )


# ==================== 主函数 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flower FedAdam联邦学习")
    parser.add_argument(
        "--mode",
        type=str,
        default="simulation",
        choices=["simulation", "server", "client"],
        help="运行模式: simulation(模拟), server(服务器), client(客户端)"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="客户端ID（仅client模式需要）"
    )
    
    args = parser.parse_args()
    
    if args.mode == "simulation":
        # 单机模拟模式
        run_simulation()
    elif args.mode == "server":
        # 分布式服务器模式
        start_server()
    elif args.mode == "client":
        # 分布式客户端模式
        start_client(args.client_id)