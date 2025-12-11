# Data Directory

This directory contains the datasets for training and evaluating the Java error classification model.

## Dataset Structure

### Files

- `train_data.json`: Training dataset (JSONL format)
- `valid_data.json`: Validation dataset (JSONL format)
- `test_data.json`: Test dataset (JSONL format)

### Data Format

Each line in the JSON files contains a single training example with the following structure:

```json
{
  "system_prompt": "Analyze the student's Java code...",
  "user_prompt": "Java code requirement: ... student code: ...",
  "feedback": "1) [Error Type] - explanation\n2) ..."
}
```

#### Fields:

- **system_prompt**: Instructions for the model on how to analyze the code
- **user_prompt**: Contains both:
  - The coding task requirements
  - The student's Java code implementation
- **feedback**: Expected output with error analysis
  - Format: Numbered list of errors
  - Each error specifies type (Syntax/Runtime/Logical) and explanation
  - If no errors: "No errors found. Code is correct."

### Error Categories

1. **Syntax Error**: Code compilation errors
   - Example: Missing semicolons, type mismatches, undefined variables

2. **Runtime Error**: Errors that occur during execution
   - Example: NullPointerException, ArrayIndexOutOfBoundsException

3. **Logical Error**: Code runs but produces incorrect results
   - Example: Wrong algorithm, off-by-one errors, incorrect conditions

## Data Statistics

| Split | Number of Samples | Description |
|-------|------------------|-------------|
| Train | ~800 samples | For model fine-tuning |
| Valid | ~100 samples | For hyperparameter tuning |
| Test | ~130 samples | For final evaluation |

## Data Preparation

To prepare data for federated learning (split into multiple clients):

```bash
python src/utils/data_preparation.py
```

This will generate:
- `data/client_0.json` - Data for client 0
- `data/client_1.json` - Data for client 1
- `data/valid.json` - Validation data (converted format)

## Usage Example

```python
import json

# Load training data
with open('data/train_data.json', 'r') as f:
    train_data = [json.loads(line) for line in f]

print(f"Number of training examples: {len(train_data)}")
print(f"First example: {train_data[0]}")
```

## Data Collection

The dataset was created by:
1. Collecting Java coding problems from educational platforms
2. Generating student solutions with various types of errors
3. Expert annotation of error types and explanations
4. Quality control and validation

## Privacy & Ethics

- All code examples are synthetic or from public educational resources
- No personal information or proprietary code is included
- Dataset is intended for research and educational purposes only

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{java_error_classification_2024,
  author = {Lei},
  title = {Java Code Error Classification Dataset},
  year = {2024},
  publisher = {GitHub}
}
```

## License

This dataset is released under the MIT License.
