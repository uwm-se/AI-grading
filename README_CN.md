# 联邦学习在LLM微调中的应用 - Java代码错误分类

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

中文文档 | [English](README.md)

## 📖 项目简介

本项目实现了一个完整的联邦学习框架,用于在Java代码错误分类任务上微调大型语言模型(LLM)。项目对比了多种联邦学习算法(FedAvg、FedProx、FedAdam)与集中式训练基线,为隐私保护的LLM微调提供了深入见解。

### 核心特性

- 🔐 **隐私保护**: 联邦学习保持训练数据去中心化
- 🤖 **多算法支持**: 支持FedAvg、FedProx和FedAdam算法
- 📊 **全面评估**: 基于GPT的自动化评分与详细指标
- 🚀 **优化训练**: LoRA + 4-bit量化实现高效微调
- 🔧 **灵活框架**: 同时支持HuggingFace和Unsloth训练管道
- 📈 **基线对比**: 集中式训练和少样本提示基线

### 任务描述

模型训练用于分析学生的Java代码并识别错误,将错误分类为:
- **语法错误(Syntax Error)**: 无法编译的代码
- **运行时错误(Runtime Error)**: 运行时崩溃的代码
- **逻辑错误(Logical Error)**: 产生错误结果的代码

## 🏗️ 项目结构

```
federated-llm-java-error-classification/
├── src/
│   ├── training/              # 训练脚本
│   │   ├── centralized_hf.py              # 集中式训练(HuggingFace)
│   │   ├── centralized_unsloth.py         # 集中式训练(Unsloth)
│   │   ├── federated_fedavg_fedprox.py    # FedAvg和FedProx算法
│   │   └── federated_fedadam.py           # FedAdam算法
│   ├── evaluation/            # 评估脚本
│   │   ├── evaluate_with_gpt.py           # 基于GPT的评估
│   │   ├── evaluate_fewshot.py            # 少样本基线
│   │   └── evaluate_unsloth.py            # Unsloth模型评估
│   └── utils/                 # 工具函数
│       └── data_preparation.py            # 数据预处理
├── scripts/                   # 训练脚本
│   ├── run_centralized.sh
│   ├── run_federated_fedadam.sh
│   ├── run_federated_fedavg.sh
│   └── run_evaluation.sh
├── data/                      # 数据集
│   ├── train_data.json        # 训练数据
│   ├── valid_data.json        # 验证数据
│   └── test_data.json         # 测试数据
├── configs/                   # 配置文件
├── docs/                      # 文档
└── examples/                  # 使用示例
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 支持CUDA的GPU(推荐: ≥16GB显存)
- PyTorch 2.0+

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/federated-llm-java-error-classification.git
cd federated-llm-java-error-classification

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows系统: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
# 为联邦学习客户端准备和分割数据
python src/utils/data_preparation.py
```

### 训练

#### 集中式训练(基线)

```bash
# 使用HuggingFace Trainer
python src/training/centralized_hf.py

# 使用Unsloth(更快、更高效)
python src/training/centralized_unsloth.py
```

#### 联邦学习

```bash
# FedAvg / FedProx (多GPU并行训练)
python src/training/federated_fedavg_fedprox.py

# FedAdam (Flower框架)
bash scripts/run_federated_fedadam.sh --mode simulation
```

### 评估

```bash
# 使用GPT评分评估训练后的模型
python src/evaluation/evaluate_with_gpt.py

# 少样本基线(无需训练)
python src/evaluation/evaluate_fewshot.py
```

## 📊 实验结果

### 性能对比

| 方法 | 测试得分(GPT-4o-mini) | 训练时间 |
|------|----------------------|----------|
| 少样本提示(0-shot) | 8.47/10 | 无 |
| 集中式(HF) | 8.89/10 | ~45分钟 |
| 集中式(Unsloth) | 8.95/10 | ~30分钟 |
| FedAvg | 8.76/10 | ~52分钟 |
| FedProx | 8.81/10 | ~54分钟 |
| FedAdam | 8.92/10 | ~48分钟 |

*基于Qwen3-4B-Base + LoRA微调的结果*

### 主要发现

- ✅ 联邦学习性能与集中式训练相当
- ✅ FedAdam在联邦算法中表现最佳
- ✅ Unsloth框架训练更快且效果更好
- ✅ 所有微调模型均显著优于少样本基线

## 🛠️ 技术细节

### 模型架构

- **基础模型**: Qwen3-4B-Base / Qwen3-8B-Base
- **微调方法**: LoRA(低秩适应)
  - Rank: 16
  - Alpha: 32
  - 目标模块: Q、K、V、O、Gate、Up、Down投影层
- **量化**: 4-bit NF4 + 嵌套量化

### 训练配置

- **批次大小**: 每设备1
- **梯度累积**: 8步(有效批次大小: 8)
- **学习率**: 1e-4(集中式), 5e-5(unsloth)
- **最大序列长度**: 1536 tokens
- **优化器**: AdamW 8-bit

### 联邦学习设置

- **客户端数量**: 2(数据均分)
- **本地训练轮次**: 3
- **通信轮次**: 1-3
- **FedProx μ**: 0.01
- **FedAdam β₁/β₂**: 0.9/0.99

## 📚 文档

详细文档请参考:

- [安装指南](docs/SETUP.md) - 安装和环境配置
- [训练指南](docs/TRAINING.md) - 详细训练说明
- [评估指南](docs/EVALUATION.md) - 评估和指标
- [系统架构](docs/ARCHITECTURE.md) - 系统架构和设计

## 🤝 贡献

欢迎贡献!请随时提交Pull Request。

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 📧 联系方式

Lei - 研究生 & 算法工程师

如有问题或讨论,请提交issue或通过邮件联系。

## 🙏 致谢

- **Flower Framework**: 提供联邦学习基础设施
- **Unsloth**: 高效的LLM微调库
- **Qwen Team**: 基础模型
- **HuggingFace**: transformers和datasets库

## 📖 引用

如果你在研究中使用了本代码,请引用:

```bibtex
@software{federated_llm_java_error,
  author = {Lei},
  title = {Federated Learning for LLM Fine-tuning on Java Error Classification},
  year = {2024},
  url = {https://github.com/yourusername/federated-llm-java-error-classification}
}
```

## 🔗 相关项目

- [Flower](https://flower.dev/) - 友好的联邦学习框架
- [Unsloth](https://github.com/unslothai/unsloth) - 高效的LLM微调
- [Qwen](https://github.com/QwenLM/Qwen) - 大型语言模型系列

---

⭐ 如果这个项目对你有帮助,请考虑给个star!
