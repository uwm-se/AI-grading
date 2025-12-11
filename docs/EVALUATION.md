# Evaluation Guide

Complete guide for evaluating trained models using various metrics and baselines.

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Methods](#evaluation-methods)
3. [Running Evaluations](#running-evaluations)
4. [Metrics](#metrics)
5. [Interpreting Results](#interpreting-results)

## Overview

The evaluation system uses **GPT-4o-mini** as an automated judge to score model outputs on multiple dimensions:

- **Error Count Accuracy**: Does the model identify the correct number of errors?
- **Error Type Classification**: Are errors correctly categorized (Syntax/Runtime/Logical)?
- **Content Quality**: Are error explanations accurate and helpful?
- **Duplication Control**: Are errors listed once without repetition?

**Overall Score**: 0-10 scale (weighted average of dimensions)

## Evaluation Methods

### 1. GPT-Based Evaluation (Primary)

Uses GPT-4o-mini to automatically score model outputs:

```bash
python src/evaluation/evaluate_with_gpt.py
```

**Scoring Rubric**:
```python
{
    "count_score": 0-10,      # Correct number of errors
    "type_score": 0-10,       # Correct error categorization  
    "content_score": 0-10,    # Quality of explanations
    "duplication_score": 0-10 # No duplicate listings
}

# Final score = average of 4 dimensions
```

### 2. Few-Shot Baseline

Evaluates the base model with few-shot prompting (no fine-tuning):

```bash
python src/evaluation/evaluate_fewshot.py
```

This provides a baseline to measure improvement from fine-tuning.

### 3. Human Evaluation (Optional)

For critical validation, manually review outputs:

```bash
# Export predictions for review
python src/evaluation/evaluate_with_gpt.py --export-predictions
```

## Running Evaluations

### Prerequisites

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create `.env` file:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Quick Evaluation

**Evaluate all methods**:
```bash
bash scripts/run_evaluation.sh --type all
```

**Evaluate specific method**:
```bash
# Centralized models only
bash scripts/run_evaluation.sh --type centralized

# Federated models only
bash scripts/run_evaluation.sh --type federated

# Few-shot baseline
bash scripts/run_evaluation.sh --type fewshot
```

### Detailed Evaluation

**Evaluate specific model**:

```python
from src.evaluation.evaluate_with_gpt import GPTScoreEvaluator

evaluator = GPTScoreEvaluator(
    base_model_name="Qwen/Qwen3-4B-Base",
    result_dir="./java_error_federated_results",
    test_data_path="./data/test_data.json",
    gpt_model="gpt-4o-mini"
)

# Evaluate FedAdam model
results = evaluator.evaluate_method("fedadam")
```

**Evaluate multiple models**:

```python
methods = ["centralized_hf", "centralized_unsloth", "fedavg", "fedprox", "fedadam"]

for method in methods:
    print(f"\n{'='*50}")
    print(f"Evaluating: {method}")
    print(f"{'='*50}")
    
    results = evaluator.evaluate_method(method)
    print(f"Overall Score: {results['overall_score']:.2f}/10")
```

### Batch Evaluation

For large-scale experiments:

```bash
# Evaluate all trained models in directory
python src/evaluation/evaluate_with_gpt.py --batch \
    --input-dir ./java_error_federated_results \
    --output-file ./evaluation_summary.json
```

## Metrics

### Primary Metric: GPT Overall Score

**Range**: 0-10
**Interpretation**:
- **9.0-10.0**: Excellent - Production ready
- **8.5-9.0**: Very Good - Minor improvements needed
- **8.0-8.5**: Good - Some refinement needed
- **7.5-8.0**: Acceptable - Significant gaps exist
- **<7.5**: Poor - Not ready for use

### Detailed Metrics

#### 1. Count Accuracy (0-10)

Measures how well the model identifies the correct number of errors.

```python
# Perfect: Identifies exactly 2 errors when there are 2
# Score: 10/10

# Partial: Identifies 1 error when there are 2
# Score: 5/10

# Incorrect: Identifies 0 or 4+ errors when there are 2
# Score: 0/10
```

#### 2. Type Classification (0-10)

Evaluates correct categorization of error types.

```python
# Perfect: All errors correctly typed as Syntax/Runtime/Logical
# Score: 10/10

# Mixed: Some correct, some incorrect types
# Score: 5-8/10

# Poor: Most types incorrect
# Score: 0-4/10
```

#### 3. Content Quality (0-10)

Assesses explanation accuracy and helpfulness.

```python
# Excellent: Clear, accurate, actionable explanations
# Score: 9-10/10

# Good: Correct but could be clearer
# Score: 7-8/10

# Acceptable: Mostly correct with some vagueness
# Score: 5-6/10

# Poor: Incorrect or unhelpful
# Score: 0-4/10
```

#### 4. Duplication Control (0-10)

Checks for repeated error listings.

```python
# Perfect: No duplicates, each error listed once
# Score: 10/10

# Minor: 1 duplicate among many errors
# Score: 8/10

# Major: Multiple duplicates
# Score: 0-5/10
```

### Secondary Metrics

Track these for deeper analysis:

```python
{
    "inference_time": 2.34,        # Seconds per sample
    "exact_match_rate": 0.45,      # % perfect predictions
    "partial_match_rate": 0.78,    # % partially correct
    "failure_rate": 0.05,          # % complete failures
    "avg_output_length": 156       # Average tokens
}
```

## Interpreting Results

### Example Output

```json
{
  "method": "fedadam",
  "overall_score": 8.92,
  "detailed_scores": {
    "count_accuracy": 8.85,
    "type_classification": 9.12,
    "content_quality": 8.76,
    "duplication_control": 9.95
  },
  "performance": {
    "exact_matches": 58,
    "partial_matches": 65,
    "failures": 7,
    "total_samples": 130
  },
  "timing": {
    "avg_inference_time": 2.1,
    "total_time": 273
  }
}
```

### Comparative Analysis

**Example Results** (Qwen3-4B-Base):

| Method | Overall | Count | Type | Content | Dedup | Time |
|--------|---------|-------|------|---------|-------|------|
| Few-shot | 8.47 | 8.2 | 8.5 | 8.4 | 9.8 | - |
| Centralized (HF) | 8.89 | 8.8 | 9.1 | 8.7 | 9.9 | 30min |
| Centralized (Unsloth) | 8.95 | 9.0 | 9.2 | 8.8 | 9.9 | 20min |
| FedAvg | 8.76 | 8.6 | 8.9 | 8.6 | 9.9 | 52min |
| FedProx | 8.81 | 8.7 | 9.0 | 8.7 | 9.9 | 54min |
| FedAdam | 8.92 | 8.9 | 9.1 | 8.8 | 9.9 | 48min |

**Key Insights**:
1. ✅ All fine-tuned models beat few-shot baseline
2. ✅ FedAdam achieves near-centralized performance
3. ✅ Unsloth provides best quality/speed tradeoff
4. ✅ Federated methods maintain privacy with <2% loss

### Statistical Significance

Test if improvements are significant:

```python
from scipy import stats

# Compare two methods
method1_scores = evaluator.get_sample_scores("centralized_hf")
method2_scores = evaluator.get_sample_scores("fedadam")

# Paired t-test
t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)

if p_value < 0.05:
    print("Difference is statistically significant")
else:
    print("Difference is not significant")
```

### Error Analysis

Analyze failure cases:

```python
# Load results
with open("evaluation_results.json", "r") as f:
    results = json.load(f)

# Find low-scoring samples
failures = [
    sample for sample in results["samples"]
    if sample["score"] < 7.0
]

print(f"Found {len(failures)} failures")

# Analyze patterns
for failure in failures[:5]:
    print(f"\nInput: {failure['input'][:100]}...")
    print(f"Expected: {failure['expected']}")
    print(f"Predicted: {failure['predicted']}")
    print(f"Score: {failure['score']}")
```

## Best Practices

### 1. Multiple Runs

Run evaluation multiple times for stability:

```bash
for i in {1..3}; do
    echo "Run $i"
    python src/evaluation/evaluate_with_gpt.py --seed $i
done
```

### 2. Cross-Validation

Evaluate on different test sets:

```python
# Split test set
test_sets = [
    "data/test_syntax.json",    # Syntax errors only
    "data/test_runtime.json",   # Runtime errors only
    "data/test_logical.json",   # Logical errors only
    "data/test_mixed.json"      # Mixed errors
]

for test_set in test_sets:
    results = evaluator.evaluate(test_data_path=test_set)
```

### 3. Cost Management

GPT evaluation can be expensive. Optimize:

```python
# Sample subset for quick checks
evaluator.evaluate(
    max_samples=50,  # Instead of all 130
    cache_responses=True  # Reuse previous GPT responses
)

# Use cheaper model for development
evaluator = GPTScoreEvaluator(
    gpt_model="gpt-3.5-turbo"  # Instead of gpt-4o-mini
)
```

### 4. Reproducibility

Ensure consistent results:

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
results = evaluator.evaluate()
```

## Troubleshooting

### API Rate Limits

If you hit OpenAI rate limits:

```python
# Add delays
import time

for sample in test_samples:
    result = evaluate_sample(sample)
    time.sleep(1)  # Wait 1 second between requests

# Or use batching
batch_size = 10
for i in range(0, len(samples), batch_size):
    batch = samples[i:i+batch_size]
    evaluate_batch(batch)
    time.sleep(60)  # Wait 1 minute between batches
```

### Inconsistent Scores

GPT responses can vary. Improve consistency:

```python
# Use temperature=0 for deterministic outputs
evaluator = GPTScoreEvaluator(
    gpt_model="gpt-4o-mini",
    temperature=0  # Most deterministic
)

# Average multiple evaluations
scores = []
for _ in range(3):
    result = evaluator.evaluate()
    scores.append(result["overall_score"])
final_score = np.mean(scores)
```

### Missing Models

If evaluation can't find models:

```bash
# Check model paths
ls -R java_error_federated_results/

# Verify model structure
tree java_error_federated_results/fedadam/final_model/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
```

## Export Results

### Generate Report

```bash
python src/evaluation/evaluate_with_gpt.py --export-report
```

Creates:
- `evaluation_report.pdf`: Formatted results
- `evaluation_summary.csv`: Tabular data
- `evaluation_plots.png`: Visualization

### Visualize Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = load_evaluation_results()

# Plot comparison
methods = list(results.keys())
scores = [results[m]["overall_score"] for m in methods]

plt.figure(figsize=(10, 6))
sns.barplot(x=methods, y=scores)
plt.ylabel("Overall Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_comparison.png")
```

---

**Last Updated**: December 2024
