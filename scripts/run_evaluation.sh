#!/bin/bash

# Model Evaluation Script
# Evaluate trained models using GPT-based scoring

set -e

echo "=========================================="
echo "Model Evaluation - Java Error Classification"
echo "=========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
EVAL_TYPE="all"  # all, centralized, federated, fewshot
GPU_ID=0
OPENAI_API_KEY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --api-key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --type TYPE          Evaluation type: all, centralized, federated, fewshot (default: all)"
            echo "  --gpu GPU_ID        GPU device ID (default: 0)"
            echo "  --api-key KEY       OpenAI API key (can also set OPENAI_API_KEY env var)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --type all --api-key YOUR_API_KEY"
            echo "  $0 --type centralized --gpu 0"
            echo "  $0 --type fewshot"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}Warning: OpenAI API key not provided${NC}"
        echo "Set via --api-key or OPENAI_API_KEY environment variable"
        echo "GPT-based evaluation will fail without API key"
    fi
else
    export OPENAI_API_KEY=$OPENAI_API_KEY
fi

echo ""
echo "Configuration:"
echo "  Evaluation Type: $EVAL_TYPE"
echo "  GPU: $GPU_ID"
echo "  API Key: ${OPENAI_API_KEY:+[SET]}${OPENAI_API_KEY:-[NOT SET]}"
echo ""

# Function to run evaluation
run_evaluation() {
    local eval_script=$1
    local eval_name=$2
    
    echo -e "${BLUE}======================================${NC}"
    echo -e "${GREEN}Evaluating: $eval_name${NC}"
    echo -e "${BLUE}======================================${NC}"
    
    if [ -f "$eval_script" ]; then
        python "$eval_script"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ $eval_name evaluation completed${NC}"
        else
            echo -e "${RED}✗ $eval_name evaluation failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ Evaluation script not found: $eval_script${NC}"
        return 1
    fi
    echo ""
}

# Run evaluations based on type
case $EVAL_TYPE in
    all)
        echo -e "${GREEN}Running all evaluations...${NC}"
        echo ""
        
        # Few-shot baseline
        run_evaluation "src/evaluation/evaluate_fewshot.py" "Few-shot Baseline"
        
        # Centralized models
        run_evaluation "src/evaluation/evaluate_with_gpt.py" "All Trained Models"
        ;;
        
    centralized)
        echo -e "${GREEN}Evaluating centralized models...${NC}"
        run_evaluation "src/evaluation/evaluate_with_gpt.py" "Centralized Models"
        ;;
        
    federated)
        echo -e "${GREEN}Evaluating federated models...${NC}"
        run_evaluation "src/evaluation/evaluate_with_gpt.py" "Federated Models"
        ;;
        
    fewshot)
        echo -e "${GREEN}Evaluating few-shot baseline...${NC}"
        run_evaluation "src/evaluation/evaluate_fewshot.py" "Few-shot Baseline"
        ;;
        
    *)
        echo -e "${RED}Unknown evaluation type: $EVAL_TYPE${NC}"
        echo "Supported types: all, centralized, federated, fewshot"
        exit 1
        ;;
esac

# Summary
echo ""
echo -e "${GREEN}=========================================="
echo "Evaluation Complete!"
echo "==========================================${NC}"
echo ""
echo "Check evaluation results in:"
echo "  - java_error_federated_results/*/evaluation_results.json"
echo "  - Console output above"
