# Project Structure

```
federated-llm-java-error-classification/
│
├── README.md                          # Main documentation (English)
├── README_CN.md                       # Chinese documentation
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contributing guidelines
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── configs/                          # Configuration files
│   └── default_config.yaml          # Default training configuration
│
├── data/                            # Dataset directory
│   ├── README.md                    # Data documentation
│   ├── train_data.json              # Training data (800 samples)
│   ├── valid_data.json              # Validation data (100 samples)
│   └── test_data.json               # Test data (130 samples)
│
├── src/                             # Source code
│   ├── __init__.py                  # Package initialization
│   │
│   ├── training/                    # Training modules
│   │   ├── __init__.py
│   │   ├── centralized_hf.py        # Centralized training (HuggingFace)
│   │   ├── centralized_unsloth.py   # Centralized training (Unsloth)
│   │   ├── federated_fedavg_fedprox.py  # FedAvg & FedProx
│   │   └── federated_fedadam.py     # FedAdam
│   │
│   ├── evaluation/                  # Evaluation modules
│   │   ├── __init__.py
│   │   ├── evaluate_with_gpt.py     # GPT-based evaluation
│   │   ├── evaluate_fewshot.py      # Few-shot baseline
│   │   └── evaluate_unsloth.py      # Unsloth model evaluation
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       └── data_preparation.py      # Data preprocessing
│
├── scripts/                         # Shell scripts
│   ├── run_centralized.sh           # Run centralized training
│   ├── run_federated_fedavg.sh      # Run FedAvg/FedProx
│   ├── run_federated_fedadam.sh     # Run FedAdam
│   └── run_evaluation.sh            # Run evaluation
│
├── docs/                            # Documentation
│   ├── SETUP.md                     # Installation guide
│   ├── TRAINING.md                  # Training guide
│   ├── EVALUATION.md                # Evaluation guide
│   └── ARCHITECTURE.md              # System architecture
│
└── examples/                        # Usage examples
    └── quick_start.ipynb            # Jupyter notebook tutorial
```

## Directory Descriptions

### Root Files

- **README.md**: Main project documentation with overview, quick start, and results
- **README_CN.md**: Chinese version of documentation
- **LICENSE**: MIT License for the project
- **CONTRIBUTING.md**: Guidelines for contributors
- **requirements.txt**: All Python package dependencies
- **.gitignore**: Files and directories to ignore in git

### configs/

Configuration files for different training scenarios:
- `default_config.yaml`: Default hyperparameters and settings

You can create custom configs by copying and modifying this file.

### data/

All dataset files:
- **train_data.json**: Training set (~800 samples)
- **valid_data.json**: Validation set (~100 samples)  
- **test_data.json**: Test set (~130 samples)
- **README.md**: Detailed data format and statistics

After running `data_preparation.py`:
- `client_0.json`, `client_1.json`: Client-specific data splits
- `valid.json`: Validation data in message format

### src/

All source code organized by functionality:

#### src/training/

Training scripts for different approaches:

1. **centralized_hf.py**
   - Standard centralized training using HuggingFace
   - Best for: Baseline comparison
   - Features: Full Trainer API, extensive logging

2. **centralized_unsloth.py**
   - Optimized centralized training using Unsloth
   - Best for: Fast experimentation
   - Features: 2-3x faster, lower memory

3. **federated_fedavg_fedprox.py**
   - Federated learning with FedAvg, FedProx
   - Best for: Multi-GPU parallel training
   - Features: True parallel execution, 4 algorithms

4. **federated_fedadam.py**
   - Federated learning with FedAdam using Flower
   - Best for: Distributed training across machines
   - Features: Client-server architecture, production-ready

#### src/evaluation/

Model evaluation scripts:

1. **evaluate_with_gpt.py**
   - Main evaluation using GPT-4o-mini as judge
   - Scores on 4 dimensions: count, type, content, dedup
   - Supports batch evaluation of multiple models

2. **evaluate_fewshot.py**
   - Baseline evaluation using few-shot prompting
   - No training required
   - Establishes lower bound for performance

3. **evaluate_unsloth.py**
   - Specialized evaluation for Unsloth-trained models
   - Handles merged LoRA adapters

#### src/utils/

Utility functions:

1. **data_preparation.py**
   - Splits training data for federated clients
   - Converts data formats
   - Generates statistics

### scripts/

Convenient shell scripts for common tasks:

1. **run_centralized.sh**
   ```bash
   # Easy centralized training
   bash scripts/run_centralized.sh --framework unsloth
   ```

2. **run_federated_fedavg.sh**
   ```bash
   # Multi-algorithm federated learning
   bash scripts/run_federated_fedavg.sh --algorithm fedprox
   ```

3. **run_federated_fedadam.sh**
   ```bash
   # Flower-based federated learning
   bash scripts/run_federated_fedadam.sh --mode simulation
   ```

4. **run_evaluation.sh**
   ```bash
   # Evaluate all models
   bash scripts/run_evaluation.sh --type all
   ```

### docs/

Comprehensive documentation:

1. **SETUP.md**
   - Installation instructions
   - Environment setup
   - Troubleshooting common issues

2. **TRAINING.md**
   - Detailed training guides
   - Hyperparameter tuning
   - Best practices

3. **EVALUATION.md**
   - Evaluation methods
   - Metric explanations
   - Result interpretation

4. **ARCHITECTURE.md**
   - System design
   - Component descriptions
   - Data flow diagrams

### examples/

Interactive examples and tutorials:

1. **quick_start.ipynb**
   - End-to-end walkthrough
   - Step-by-step instructions
   - Visualization examples

## File Naming Conventions

### Python Files

- `centralized_*.py`: Centralized training approaches
- `federated_*.py`: Federated learning approaches
- `evaluate_*.py`: Evaluation scripts
- `*_preparation.py`: Data preprocessing utilities

### Shell Scripts

- `run_*.sh`: Executable scripts for training/evaluation
- All scripts accept `--help` flag for usage information

### Documentation

- `*.md`: Markdown documentation
- `README*.md`: Project overview and guides
- `UPPERCASE.md`: Important project-level docs

## Output Directories (Created During Training)

```
java_error_federated_results/      # Main results directory
├── centralized_hf/               # HuggingFace centralized results
│   ├── checkpoints/             # Training checkpoints
│   ├── final_model/             # Final trained model
│   ├── logs/                    # TensorBoard logs
│   ├── config.json              # Training configuration
│   └── training_history.json   # Training metrics
│
├── fedavg/                      # FedAvg results
├── fedprox/                     # FedProx results
├── fedadam/                     # FedAdam results
└── ...
```

## Quick Navigation

**Want to...**

- **Start training?** → See `docs/TRAINING.md`
- **Set up environment?** → See `docs/SETUP.md`
- **Evaluate models?** → See `docs/EVALUATION.md`
- **Understand architecture?** → See `docs/ARCHITECTURE.md`
- **Contribute?** → See `CONTRIBUTING.md`
- **Quick demo?** → See `examples/quick_start.ipynb`

## File Count Summary

- **Python files**: 11 (training, evaluation, utils)
- **Shell scripts**: 4 (automation)
- **Documentation**: 8 (guides and references)
- **Data files**: 3 (train, valid, test)
- **Config files**: 1 (YAML)
- **Examples**: 1 (Jupyter notebook)

**Total**: 28 key files in organized structure

---

For more details, see the main [README.md](../README.md)
