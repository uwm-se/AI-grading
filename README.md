# Federated Learning for LLM Fine-tuning on Java Error Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文文档](README_CN.md) | English

## 📖 Overview

This repository implements a comprehensive federated learning framework for fine-tuning Large Language Models (LLMs) on Java code error classification tasks. The project compares various federated learning algorithms (FedAvg, FedProx, FedAdam) against centralized training baselines, providing insights into privacy-preserving LLM fine-tuning.

### Key Features

- 🔐 **Privacy-Preserving**: Federated learning keeps training data decentralized
- 🤖 **Multiple Algorithms**: Supports FedAvg, FedProx, and FedAdam
- 📊 **Comprehensive Evaluation**: GPT-based automated scoring with detailed metrics
- 🚀 **Optimized Training**: LoRA + 4-bit quantization for efficient fine-tuning
- 🔧 **Flexible Framework**: Both HuggingFace and Unsloth training pipelines
- 📈 **Baseline Comparison**: Centralized training and few-shot prompting baselines

### Task Description

The model is trained to analyze Java student code and identify errors, categorizing them as:
- **Syntax Errors**: Code that won't compile
- **Runtime Errors**: Code that crashes during execution
- **Logical Errors**: Code that produces incorrect results

## 🏗️ Project Structure

```
federated-llm-java-error-classification/
├── src/
│   ├── training/              # Training scripts
│   │   ├── centralized_hf.py              # Centralized training (HuggingFace)
│   │   ├── centralized_unsloth.py         # Centralized training (Unsloth)
│   │   ├── federated_fedavg_fedprox.py    # FedAvg & FedProx algorithms
│   │   └── federated_fedadam.py           # FedAdam algorithm
│   ├── evaluation/            # Evaluation scripts
│   │   ├── evaluate_with_gpt.py           # GPT-based evaluation
│   │   ├── evaluate_fewshot.py            # Few-shot baseline
│   │   └── evaluate_unsloth.py            # Unsloth model evaluation
│   └── utils/                 # Utility functions
│       └── data_preparation.py            # Data preprocessing
├── scripts/                   # Shell scripts for training
│   ├── run_centralized.sh
│   ├── run_federated_fedadam.sh
│   ├── run_federated_fedavg.sh
│   └── run_evaluation.sh
├── data/                      # Datasets
│   ├── train_data.json        # Training data
│   ├── valid_data.json        # Validation data
│   └── test_data.json         # Test data
├── configs/                   # Configuration files
├── docs/                      # Documentation
└── examples/                  # Usage examples
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: ≥16GB VRAM)
- PyTorch 2.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-llm-java-error-classification.git
cd federated-llm-java-error-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Prepare and split data for federated clients
python src/utils/data_preparation.py
```

### Training

#### Centralized Training (Baseline)

```bash
# Using HuggingFace Trainer
python src/training/centralized_hf.py

# Using Unsloth (faster, more memory-efficient)
python src/training/centralized_unsloth.py
```

#### Federated Learning

```bash
# FedAvg / FedProx (Multi-GPU parallel training)
python src/training/federated_fedavg_fedprox.py

# FedAdam (Flower framework)
bash scripts/run_federated_fedadam.sh --mode simulation
```

### Evaluation

```bash
# Evaluate trained models with GPT scoring
python src/evaluation/evaluate_with_gpt.py

# Few-shot baseline (no training)
python src/evaluation/evaluate_fewshot.py
```

## 📊 Experimental Results

### Performance Comparison

| Method | Error Count Accuracy | Error Type Precision | Explanation Quality |
|--------|---------------------|---------------------|-------------------|
| Base Model | 3.28 | 3.84 | 3.12 |
| Centralized | 8.53 | 4.91 | 6.0 |
| FedAvg | 7.8 | 4.67 | 5.73 |
| FedProx | 7.58 | 4.26 | 5.26 |
| FedAdam | 6.23 | 5.21 | 4.97 |

*Results based on Qwen3-4B-Base with LoRA fine-tuning*

### Key Findings

- ✅ Federated learning achieves comparable performance to centralized training
- ✅ FedAdam shows the best performance among federated algorithms
- ✅ Unsloth framework provides faster training with better results
- ✅ All fine-tuned models significantly outperform few-shot baseline

## 🛠️ Technical Details

### Model Architecture

- **Base Model**: Qwen3-4B-Base / Qwen3-8B-Base
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: Q, K, V, O, Gate, Up, Down projections
- **Quantization**: 4-bit NF4 with nested quantization

### Training Configuration

- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps (effective batch size: 8)
- **Learning Rate**: 1e-4 (centralized), 5e-5 (unsloth)
- **Max Sequence Length**: 1536 tokens
- **Optimizer**: AdamW 8-bit

### Federated Learning Settings

- **Clients**: 2 (data split evenly)
- **Local Epochs**: 3
- **Communication Rounds**: 1-3
- **FedProx μ**: 0.01
- **FedAdam β₁/β₂**: 0.9/0.99

## 📚 Documentation

For detailed documentation, please refer to:

- [Setup Guide](docs/SETUP.md) - Installation and environment setup
- [Training Guide](docs/TRAINING.md) - Detailed training instructions
- [Evaluation Guide](docs/EVALUATION.md) - Evaluation and metrics
- [Architecture](docs/ARCHITECTURE.md) - System architecture and design

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Lei - Graduate Student & Algorithm Engineer

For questions and discussions, please open an issue or contact via email.

## 🙏 Acknowledgments

- **Flower Framework**: For providing federated learning infrastructure
- **Unsloth**: For efficient LLM fine-tuning library
- **Qwen Team**: For the base model
- **HuggingFace**: For transformers and datasets libraries

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yao2026java,
  author    = {Lei Yao and Hiba Alsghaier and Tian Zhao},
  title     = {Enhancing Java Grading Through Large Language Models},
  booktitle = {Proceedings of the 41st ACM/SIGAPP Symposium on Applied Computing (SAC '26)},
  year      = {2026},
  pages     = {3},
  address   = {Thessaloniki, Greece},
  publisher = {ACM},
  doi       = {10.1145/3748522.3780023},
  url       = {https://doi.org/10.1145/3748522.3780023}
}
```

## 🔗 Related Projects

- [Flower](https://flower.dev/) - A friendly federated learning framework
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient LLM fine-tuning
- [Qwen](https://github.com/QwenLM/Qwen) - Large language model series

---

⭐ If you find this project helpful, please consider giving it a star!
