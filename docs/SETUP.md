# Setup Guide

This guide will help you set up the environment and dependencies for the Federated Learning LLM Fine-tuning project.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Dependency Installation](#dependency-installation)
4. [Data Preparation](#data-preparation)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended: ≥16GB VRAM)
  - Minimum: RTX 3090 / A5000 (24GB)
  - Recommended: RTX 4090 / A6000 (48GB) or better
- **RAM**: ≥32GB system memory
- **Storage**: ≥100GB free space (for models and data)

### Software

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA**: 11.8+ or 12.1+ (for GPU acceleration)
- **Git**: For cloning the repository

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/federated-llm-java-error-classification.git
cd federated-llm-java-error-classification
```

### 2. Create Virtual Environment

We recommend using `venv` or `conda` for environment management.

#### Using venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using conda

```bash
conda create -n fed-llm python=3.10
conda activate fed-llm
```

### 3. Verify CUDA Installation

```bash
nvidia-smi  # Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

## Dependency Installation

### Basic Installation

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Optional: Install Unsloth (for faster training)

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

### Verify Installation

```bash
python -c "
import torch
import transformers
import peft
import flwr
print('✓ All core dependencies installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

## Data Preparation

### 1. Check Data Files

Ensure your data files are in the `data/` directory:

```bash
ls -lh data/
# Should show:
# - train_data.json
# - valid_data.json
# - test_data.json
```

### 2. Prepare Data for Federated Learning

Split the training data into multiple clients:

```bash
python src/utils/data_preparation.py
```

This will create:
- `data/client_0.json`
- `data/client_1.json`
- `data/valid.json` (converted format)

### 3. Verify Data Format

```bash
python -c "
import json
with open('data/train_data.json', 'r') as f:
    data = [json.loads(line) for line in f]
print(f'✓ Training samples: {len(data)}')
print(f'✓ Sample keys: {list(data[0].keys())}')
"
```

## Configuration

### 1. Model Configuration

Edit the configuration in training scripts or create a config file:

```python
# Example configuration
MODEL_NAME = "Qwen/Qwen3-4B-Base"  # or Qwen3-8B-Base
LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1536
```

### 2. Federated Learning Configuration

```python
NUM_CLIENTS = 2
NUM_ROUNDS = 3
LOCAL_EPOCHS = 3
ALGORITHM = "fedavg"  # fedavg, fedprox, fedadam
```

### 3. OpenAI API Key (for evaluation)

Set your OpenAI API key for GPT-based evaluation:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. **Reduce batch size**: Set `BATCH_SIZE = 1`
2. **Increase gradient accumulation**: Set `GRADIENT_ACCUMULATION_STEPS = 16`
3. **Enable gradient checkpointing**: Set `USE_GRADIENT_CHECKPOINTING = True`
4. **Use smaller model**: Switch from 8B to 4B model
5. **Use Unsloth**: More memory-efficient than standard HuggingFace

```python
# Memory-efficient configuration
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
USE_GRADIENT_CHECKPOINTING = True
USE_4BIT = True
USE_NESTED_QUANT = True
```

### Import Errors

If you get import errors:

```bash
# Reinstall specific packages
pip install --upgrade transformers datasets peft

# Check conflicting versions
pip list | grep -E "torch|transformers|peft"
```

### Slow Training

Optimize training speed:

1. **Use Unsloth**: 2-3x faster than HuggingFace
2. **Enable bfloat16**: Faster on Ampere+ GPUs
3. **Optimize data loading**: Increase `num_workers`
4. **Use compiled models**: PyTorch 2.0+ compilation

```python
# Speed optimization
USE_NESTED_QUANT = True
DATALOADER_NUM_WORKERS = 4
GRADIENT_CHECKPOINTING = "unsloth"  # If using Unsloth
```

### Flower Connection Issues

For distributed federated learning:

```bash
# Check if port is available
lsof -i :8080

# Kill existing processes
pkill -f "python.*federated"

# Increase timeout
export FLOWER_TIMEOUT=300
```

### Model Download Issues

If model download fails:

```bash
# Set mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir ./models/qwen3-4b

# Then update MODEL_NAME in scripts
MODEL_NAME = "./models/qwen3-4b"
```

## Next Steps

After successful setup:

1. **Train a baseline model**: See [TRAINING.md](TRAINING.md)
2. **Run federated learning**: See [TRAINING.md](TRAINING.md#federated-learning)
3. **Evaluate models**: See [EVALUATION.md](EVALUATION.md)

## Getting Help

If you encounter issues not covered here:

1. Check existing [GitHub Issues](https://github.com/yourusername/federated-llm-java-error-classification/issues)
2. Create a new issue with:
   - Error message
   - System configuration
   - Steps to reproduce
3. Contact the maintainer

---

**Last Updated**: December 2024
