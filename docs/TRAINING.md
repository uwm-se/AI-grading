# Training Guide

Complete guide for training models using both centralized and federated learning approaches.

## Table of Contents

1. [Overview](#overview)
2. [Centralized Training](#centralized-training)
3. [Federated Learning](#federated-learning)
4. [Advanced Configuration](#advanced-configuration)
5. [Monitoring Training](#monitoring-training)

## Overview

This project supports multiple training paradigms:

| Method | Description | Use Case |
|--------|-------------|----------|
| **Centralized (HF)** | Standard HuggingFace training | Baseline comparison |
| **Centralized (Unsloth)** | Optimized training with Unsloth | Faster baseline |
| **FedAvg** | Federated Averaging | Privacy-preserving, basic FL |
| **FedProx** | FedAvg + proximal term | Heterogeneous data |
| **FedAdam** | Adaptive optimization | Better convergence |

## Centralized Training

Centralized training uses all data in one location (baseline for comparison).

### Quick Start

```bash
# Using HuggingFace Trainer
bash scripts/run_centralized.sh --framework hf

# Using Unsloth (faster, recommended)
bash scripts/run_centralized.sh --framework unsloth --gpu 0
```

### Manual Training

#### HuggingFace Framework

```bash
python src/training/centralized_hf.py
```

**Key Configuration** (in `centralized_hf.py`):

```python
class CentralizedConfig:
    MODEL_NAME = "Qwen/Qwen3-8B-Base"
    LORA_R = 16
    LORA_ALPHA = 32
    NUM_EPOCHS = 3
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1e-4
    MAX_SEQ_LENGTH = 1536
```

#### Unsloth Framework

```bash
python src/training/centralized_unsloth.py
```

**Advantages of Unsloth**:
- 2-3x faster training
- Lower memory usage
- Better performance (8.95 vs 8.89 score)
- More efficient gradient checkpointing

### Expected Output

```
Training Configuration:
  Training samples: 800
  Validation samples: 100
  Epochs: 3
  Batch size: 1
  Effective batch size: 8
  
Training Progress:
  Epoch 1/3: 100%|████████| 100/100 [15:00<00:00]
  train_loss: 0.245, eval_loss: 0.198
  
Model saved to: ./qwen3_java_evaluator_lora/final_model
```

### Training Time

| Model | Framework | Time (RTX 4090) | VRAM Usage |
|-------|-----------|-----------------|------------|
| Qwen3-4B | HF | ~30 min | ~18GB |
| Qwen3-4B | Unsloth | ~20 min | ~14GB |
| Qwen3-8B | HF | ~45 min | ~22GB |
| Qwen3-8B | Unsloth | ~30 min | ~18GB |

## Federated Learning

Federated learning trains models across distributed clients while keeping data private.

### Data Preparation

First, split data for multiple clients:

```bash
python src/utils/data_preparation.py
```

This creates:
- `data/client_0.json` (50% of data)
- `data/client_1.json` (50% of data)
- `data/valid.json` (validation set)

### FedAvg & FedProx (Multi-GPU)

**Single Command**:

```bash
# FedAvg
bash scripts/run_federated_fedavg.sh --algorithm fedavg --clients 2 --rounds 3

# FedProx (better for heterogeneous data)
bash scripts/run_federated_fedavg.sh --algorithm fedprox --clients 2 --rounds 3
```

**Manual Execution**:

```bash
python src/training/federated_fedavg_fedprox.py
```

**Key Features**:
- Multi-GPU parallel training
- Clients train simultaneously
- Automatic gradient aggregation
- Support for 2-8 clients

### FedAdam (Flower Framework)

**Single Command**:

```bash
bash scripts/run_federated_fedadam.sh --mode simulation
```

**Distributed Mode** (multiple machines):

```bash
# On server machine
bash scripts/run_federated_fedadam.sh --mode server

# On client machines
bash scripts/run_federated_fedadam.sh --mode client --client-id 0
bash scripts/run_federated_fedadam.sh --mode client --client-id 1
```

**Algorithm Configuration**:

```python
# In federated_fedadam.py
class FederatedConfig:
    NUM_CLIENTS = 2
    NUM_ROUNDS = 3
    LOCAL_EPOCHS = 3
    
    # FedAdam hyperparameters
    FEDADAM_BETA1 = 0.9   # First moment
    FEDADAM_BETA2 = 0.99  # Second moment
    FEDADAM_TAU = 1e-3    # Learning rate
```

### Training Workflow

1. **Round 1**: Each client trains locally
   ```
   Client 0: Train on client_0.json (3 epochs)
   Client 1: Train on client_1.json (3 epochs)
   ```

2. **Aggregation**: Server aggregates gradients
   ```
   FedAvg: w_global = Σ(w_i * n_i) / Σ(n_i)
   FedProx: w_global = FedAvg + μ||w - w_old||²
   FedAdam: w_global = w - τ * m_t / (√v_t + ε)
   ```

3. **Broadcast**: Updated model sent to clients

4. **Repeat**: Steps 1-3 for NUM_ROUNDS

### Expected Output

```
Federated Learning Configuration:
  Algorithm: FedAdam
  Clients: 2
  Rounds: 3
  Local epochs: 3

Round 1/3:
  Client 0: train_loss=0.312, time=8.5min
  Client 1: train_loss=0.298, time=8.3min
  Global eval_loss=0.245

Round 2/3:
  Client 0: train_loss=0.245, time=8.2min
  Client 1: train_loss=0.238, time=8.1min
  Global eval_loss=0.198

Model saved: java_error_federated_results/fedadam/final_model
```

## Advanced Configuration

### Hyperparameter Tuning

**Learning Rate**:
```python
# Conservative (safer)
LEARNING_RATE = 5e-5

# Standard
LEARNING_RATE = 1e-4

# Aggressive (faster but may diverge)
LEARNING_RATE = 2e-4
```

**LoRA Configuration**:
```python
# Smaller model (faster, less capacity)
LORA_R = 8
LORA_ALPHA = 16

# Standard
LORA_R = 16
LORA_ALPHA = 32

# Larger model (slower, more capacity)
LORA_R = 32
LORA_ALPHA = 64
```

**Federated Learning Parameters**:

```python
# More communication, better convergence
NUM_ROUNDS = 5
LOCAL_EPOCHS = 2

# Less communication, faster but may underfit
NUM_ROUNDS = 1
LOCAL_EPOCHS = 5

# FedProx regularization (for heterogeneous data)
FEDPROX_MU = 0.01  # Increase to 0.1 for more regularization
```

### Memory Optimization

If you encounter OOM errors:

```python
# Minimal memory configuration
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
USE_GRADIENT_CHECKPOINTING = True
USE_4BIT = True
USE_NESTED_QUANT = True
MAX_SEQ_LENGTH = 1024  # Reduce from 1536

# For Unsloth
USE_GRADIENT_CHECKPOINTING = "unsloth"
```

### Multi-GPU Training

**For FedAvg/FedProx** (automatic):
```python
NUM_CLIENTS = 4
NUM_GPUS = 2  # Clients 0,1 on GPU 0; Clients 2,3 on GPU 1
```

**For centralized training**:
```python
# Single GPU (default)
CUDA_VISIBLE_DEVICES=0 python src/training/centralized_hf.py

# Multi-GPU (data parallel)
CUDA_VISIBLE_DEVICES=0,1 python src/training/centralized_hf.py
```

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./java_error_federated_results/logs

# View at http://localhost:6006
```

### Weights & Biases (Optional)

```python
# In training script
import wandb

wandb.init(
    project="federated-llm",
    config={
        "model": MODEL_NAME,
        "algorithm": "fedadam",
        "learning_rate": LEARNING_RATE,
    }
)

# Trainer will automatically log to wandb
training_args = TrainingArguments(
    ...
    report_to="wandb"
)
```

### Logging

Check training logs:

```bash
# View real-time logs
tail -f java_error_federated_results/fedadam/training.log

# Check metrics
cat java_error_federated_results/fedadam/training_history.json
```

### GPU Monitoring

Monitor GPU usage during training:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

## Troubleshooting

### Training Hangs

**Problem**: Training stops responding

**Solutions**:
```bash
# Check for deadlocks
ps aux | grep python

# Kill hanging processes
pkill -9 -f "training"

# Restart with single client to isolate issue
NUM_CLIENTS=1 python src/training/federated_fedavg_fedprox.py
```

### Diverging Loss

**Problem**: Loss increases or becomes NaN

**Solutions**:
```python
# Reduce learning rate
LEARNING_RATE = 5e-5  # Down from 1e-4

# Add gradient clipping
MAX_GRAD_NORM = 1.0

# Increase warmup
WARMUP_RATIO = 0.1  # Up from 0.03
```

### Poor Convergence

**Problem**: Loss plateaus at high value

**Solutions**:
1. Train longer: Increase `NUM_ROUNDS` or `LOCAL_EPOCHS`
2. Use adaptive optimizer: Switch to FedAdam
3. Check data quality: Verify data preparation
4. Increase model capacity: Use `LORA_R = 32`

## Next Steps

After training:

1. **Evaluate models**: See [EVALUATION.md](EVALUATION.md)
2. **Compare results**: Analyze training curves and metrics
3. **Fine-tune hyperparameters**: Based on validation loss
4. **Deploy best model**: Save and export for inference

---

**Last Updated**: December 2024
