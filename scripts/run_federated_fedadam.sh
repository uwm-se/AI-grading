#!/bin/bash
#
# Flower FedAdam 联邦学习运行脚本
# 版本: v1.0
#

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
SCRIPT_PATH="fedadam_train.py"
SERVER_ADDRESS="localhost:8080"
NUM_CLIENTS=2

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo ""
    echo "=========================================="
    echo "  Flower FedAdam 联邦学习运行脚本"
    echo "=========================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --mode MODE       运行模式 (simulation/server/client/distributed)"
    echo "                        - simulation: 单机模拟模式 (默认)"
    echo "                        - server: 启动服务器"
    echo "                        - client: 启动单个客户端"
    echo "                        - distributed: 启动服务器+所有客户端"
    echo "  -c, --client-id ID    客户端ID (仅client模式需要)"
    echo "  -n, --num-clients N   客户端数量 (默认: 2)"
    echo "  -h, --help            显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                           # 单机模拟模式"
    echo "  $0 -m simulation             # 单机模拟模式"
    echo "  $0 -m server                 # 启动服务器"
    echo "  $0 -m client -c 0            # 启动客户端0"
    echo "  $0 -m distributed -n 2       # 启动服务器+2个客户端"
    echo ""
}

# 检查Python环境
check_environment() {
    print_info "检查运行环境..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "未找到Python，请先安装Python"
        exit 1
    fi
    
    # 检查脚本文件
    if [ ! -f "$SCRIPT_PATH" ]; then
        print_error "未找到训练脚本: $SCRIPT_PATH"
        exit 1
    fi
    
    # 检查必要的库
    python -c "import flwr" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "未安装Flower库，正在安装..."
        pip install flwr --break-system-packages
    fi
    
    # 检查GPU
    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "未检测到GPU或CUDA不可用"
        exit 1
    fi
    
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    print_success "检测到 $GPU_COUNT 块GPU"
    
    # 检查数据目录
    if [ ! -d "./data" ]; then
        print_warning "未找到数据目录 ./data"
    fi
}

# 运行模拟模式
run_simulation() {
    print_info "启动单机模拟模式..."
    echo ""
    echo "=========================================="
    echo "  模式: Simulation (单机模拟)"
    echo "  客户端数量: $NUM_CLIENTS"
    echo "=========================================="
    echo ""
    
    python "$SCRIPT_PATH" --mode simulation
    
    if [ $? -eq 0 ]; then
        print_success "训练完成！"
    else
        print_error "训练失败"
        exit 1
    fi
}

# 启动服务器
run_server() {
    print_info "启动Flower服务器..."
    echo ""
    echo "=========================================="
    echo "  模式: Server"
    echo "  地址: $SERVER_ADDRESS"
    echo "=========================================="
    echo ""
    
    python "$SCRIPT_PATH" --mode server
}

# 启动客户端
run_client() {
    local client_id=$1
    
    print_info "启动客户端 $client_id..."
    echo ""
    echo "=========================================="
    echo "  模式: Client"
    echo "  客户端ID: $client_id"
    echo "  服务器地址: $SERVER_ADDRESS"
    echo "=========================================="
    echo ""
    
    python "$SCRIPT_PATH" --mode client --client-id "$client_id"
}

# 分布式模式：启动服务器和所有客户端
run_distributed() {
    print_info "启动分布式训练..."
    echo ""
    echo "=========================================="
    echo "  模式: Distributed"
    echo "  服务器地址: $SERVER_ADDRESS"
    echo "  客户端数量: $NUM_CLIENTS"
    echo "=========================================="
    echo ""
    
    # 启动服务器（后台运行）
    print_info "启动服务器..."
    python "$SCRIPT_PATH" --mode server &
    SERVER_PID=$!
    
    # 等待服务器启动
    sleep 5
    
    # 检查服务器是否启动成功
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        print_error "服务器启动失败"
        exit 1
    fi
    print_success "服务器已启动 (PID: $SERVER_PID)"
    
    # 启动所有客户端
    CLIENT_PIDS=()
    for ((i=0; i<NUM_CLIENTS; i++)); do
        print_info "启动客户端 $i..."
        
        # 分配GPU
        GPU_ID=$((i % $(python -c "import torch; print(torch.cuda.device_count())")))
        CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_PATH" --mode client --client-id $i &
        CLIENT_PIDS+=($!)
        
        sleep 2
    done
    
    print_success "所有客户端已启动"
    echo ""
    print_info "等待训练完成..."
    print_info "按 Ctrl+C 可终止所有进程"
    
    # 等待服务器完成
    wait $SERVER_PID
    
    # 清理客户端进程
    for pid in "${CLIENT_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid
        fi
    done
    
    print_success "分布式训练完成！"
}

# 清理函数
cleanup() {
    print_warning "接收到中断信号，正在清理..."
    
    # 终止所有子进程
    pkill -P $$
    
    print_info "已终止所有进程"
    exit 0
}

# 注册清理函数
trap cleanup SIGINT SIGTERM

# 解析命令行参数
MODE="simulation"
CLIENT_ID=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -c|--client-id)
            CLIENT_ID="$2"
            shift 2
            ;;
        -n|--num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主逻辑
echo ""
echo "========================================"
echo "  Flower FedAdam 联邦学习"
echo "========================================"
echo ""

check_environment

case $MODE in
    simulation)
        run_simulation
        ;;
    server)
        run_server
        ;;
    client)
        run_client "$CLIENT_ID"
        ;;
    distributed)
        run_distributed
        ;;
    *)
        print_error "未知模式: $MODE"
        show_help
        exit 1
        ;;
esac