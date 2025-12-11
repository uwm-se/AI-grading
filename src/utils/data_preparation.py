"""
数据预处理：将训练数据分配给多个客户端
"""

import json
import os
import random
from typing import List, Dict

def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 尝试作为JSONL格式读取（每行一个JSON）
        first_line = f.readline()
        f.seek(0)
        
        try:
            # 检查是否是JSONL格式
            json.loads(first_line)
            # 是JSONL格式
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
            return data
        except:
            # 不是JSONL，尝试作为标准JSON数组
            f.seek(0)
            return json.load(f)


def convert_to_messages_format(data_item: Dict) -> Dict:
    """转换为messages格式"""
    messages = [
        {
            "role": "system",
            "content": data_item["system_prompt"]
        },
        {
            "role": "user",
            "content": data_item["user_prompt"]
        },
        {
            "role": "assistant",
            "content": data_item["feedback"]
        }
    ]
    return {"messages": messages}


def split_data_for_clients(
    train_data: List[Dict],
    num_clients: int = 2,
    random_seed: int = 42
) -> List[List[Dict]]:
    """将训练数据分配给多个客户端"""
    
    random.seed(random_seed)
    
    # 打乱数据
    shuffled_data = train_data.copy()
    random.shuffle(shuffled_data)
    
    # 分配数据
    client_data = [[] for _ in range(num_clients)]
    
    for idx, item in enumerate(shuffled_data):
        client_idx = idx % num_clients
        client_data[client_idx].append(item)
    
    return client_data


def main():
    """主函数"""
    
    print("="*70)
    print("数据预处理：分配客户端数据")
    print("="*70)
    
    # 配置
    NUM_CLIENTS = 2
    DATA_DIR = "./data"
    
    # 创建输出目录
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 加载训练数据
    print("\n加载训练数据...")
    train_data = load_json_data("./data/new_train_data.json")
    train_data = train_data
    print(f"✓ 训练数据: {len(train_data)} 样本")
    
    # 加载验证数据
    print("\n加载验证数据...")
    valid_data = load_json_data("./data/valid_data.json")
    print(f"✓ 验证数据: {len(valid_data)} 样本")
    
    # 转换格式
    print("\n转换数据格式...")
    train_data_formatted = [convert_to_messages_format(item) for item in train_data]
    valid_data_formatted = [convert_to_messages_format(item) for item in valid_data]
    
    # 分配客户端数据
    print(f"\n分配数据给 {NUM_CLIENTS} 个客户端...")
    client_datasets = split_data_for_clients(train_data_formatted, NUM_CLIENTS)
    
    # 保存客户端数据
    for client_id, client_data in enumerate(client_datasets):
        output_file = f"{DATA_DIR}/client_{client_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(client_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 客户端 {client_id}: {len(client_data)} 样本 -> {output_file}")
    
    # 保存验证数据
    valid_output_file = f"{DATA_DIR}/valid.json"
    with open(valid_output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data_formatted, f, ensure_ascii=False, indent=2)
    print(f"✓ 验证集: {len(valid_data_formatted)} 样本 -> {valid_output_file}")
    
    # 统计信息
    print("\n" + "="*70)
    print("数据统计")
    print("="*70)
    print(f"总训练样本: {len(train_data_formatted)}")
    print(f"总验证样本: {len(valid_data_formatted)}")
    print(f"\n客户端数据分布:")
    for client_id, client_data in enumerate(client_datasets):
        percentage = len(client_data) / len(train_data_formatted) * 100
        print(f"  客户端 {client_id}: {len(client_data)} 样本 ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print("数据预处理完成！")
    print("="*70)
    print(f"\n数据保存在: {DATA_DIR}/")
    print("\n下一步: 运行训练脚本")
    print("  python federated_training.py")
    print("="*70)


if __name__ == "__main__":
    main()
