#!/bin/bash

# Federated Learning Training Script - FedAvg & FedProx
# Multi-GPU parallel training for multiple clients

set -e

echo "=========================================="
echo "Federated Learning: FedAvg & FedProx"
echo "=========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
ALGORITHM="fedavg"  # fedavg, fedprox, fedadam
NUM_CLIENTS=2
NUM_ROUNDS=1
GPU_IDS="0,1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --algorithm ALGO     Federated algorithm: fedavg, fedprox, fedadam (default: fedavg)"
            echo "  --clients NUM        Number of clients (default: 2)"
            echo "  --rounds NUM         Communication rounds (default: 1)"
            echo "  --gpus GPU_IDS       GPU IDs separated by comma (default: 0,1)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --algorithm fedavg --clients 2 --rounds 3"
            echo "  $0 --algorithm fedprox --gpus 0,1,2,3 --clients 4"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Clients: $NUM_CLIENTS"
echo "  Rounds: $NUM_ROUNDS"
echo "  GPUs: $GPU_IDS"
echo ""

# Check data preparation
if [ ! -f "data/client_0.json" ]; then
    echo -e "${YELLOW}Data not prepared for federated learning.${NC}"
    echo "Running data preparation..."
    python src/utils/data_preparation.py
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Check GPU availability
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
echo -e "${GREEN}Detected $NUM_GPUS GPUs${NC}"

if [ $NUM_GPUS -lt $NUM_CLIENTS ]; then
    echo -e "${YELLOW}Warning: Number of GPUs ($NUM_GPUS) < Number of clients ($NUM_CLIENTS)${NC}"
    echo "Clients will share GPUs"
fi

# Run federated training
echo ""
echo -e "${GREEN}Starting federated training...${NC}"
echo ""

# Modify script based on algorithm
case $ALGORITHM in
    fedavg|fedprox)
        # Use multi-GPU script
        python src/training/federated_fedavg_fedprox.py \
            --algorithm "$ALGORITHM" \
            --num-clients $NUM_CLIENTS \
            --num-rounds $NUM_ROUNDS
        ;;
    fedadam)
        # Use FedAdam-specific script
        python src/training/federated_fedadam.py \
            --mode simulation \
            --num-clients $NUM_CLIENTS \
            --num-rounds $NUM_ROUNDS
        ;;
    *)
        echo -e "${RED}Unknown algorithm: $ALGORITHM${NC}"
        exit 1
        ;;
esac

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Federated training completed successfully!"
    echo "==========================================${NC}"
    echo ""
    echo "Results saved in: java_error_federated_results/$ALGORITHM/"
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Federated training failed!"
    echo "==========================================${NC}"
    exit 1
fi
