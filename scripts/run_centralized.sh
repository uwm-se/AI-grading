#!/bin/bash

# Centralized Training Script
# Train models using centralized learning as baseline

set -e  # Exit on error

echo "=========================================="
echo "Centralized Training for Java Error Classification"
echo "=========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
FRAMEWORK="hf"  # hf or unsloth
GPU_ID=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --framework FRAMEWORK    Training framework: hf (HuggingFace) or unsloth (default: hf)"
            echo "  --gpu GPU_ID            GPU device ID (default: 0)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --framework hf          # Train with HuggingFace"
            echo "  $0 --framework unsloth     # Train with Unsloth (faster)"
            echo "  $0 --framework hf --gpu 1  # Train on GPU 1"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "Configuration:"
echo "  Framework: $FRAMEWORK"
echo "  GPU: $GPU_ID"
echo ""

# Check if data exists
if [ ! -f "data/train_data.json" ]; then
    echo -e "${RED}Error: Training data not found!${NC}"
    echo "Please ensure data/train_data.json exists"
    exit 1
fi

# Run training based on framework
if [ "$FRAMEWORK" = "hf" ]; then
    echo -e "${GREEN}Starting centralized training with HuggingFace...${NC}"
    python src/training/centralized_hf.py
    
elif [ "$FRAMEWORK" = "unsloth" ]; then
    echo -e "${GREEN}Starting centralized training with Unsloth...${NC}"
    python src/training/centralized_unsloth.py
    
else
    echo -e "${RED}Error: Unknown framework '$FRAMEWORK'${NC}"
    echo "Supported frameworks: hf, unsloth"
    exit 1
fi

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Training completed successfully!"
    echo "==========================================${NC}"
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Training failed!"
    echo "==========================================${NC}"
    exit 1
fi
