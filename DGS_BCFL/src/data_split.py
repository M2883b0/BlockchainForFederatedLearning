# -*- coding:utf-8 -*-
# @FileName :data_split.py
# @Time :2025/10/7 21:19
# @Author :M2883b0


"""
- 区块链联邦学习系统 -
  联邦学习数据提取器

  本模块实现了MNIST数据集的获取、分割和持久化存储功能，包括：
  1. 从PyTorch获取MNIST数据集
  2. 数据集的分割，用于联邦学习场景
  3. 数据的保存和加载
  4. 数据集信息显示
"""

import torch
from torchvision import datasets, transforms
from DGS_BCFL.src.utils.logger import setup_logger, info, debug, warning, error


def get_mnist_pytorch_dataset(train=True, transform=None):
    """
    获取PyTorch格式的MNIST数据集对象

    参数:
        train: 是否返回训练数据集，True为训练集，False为测试集
        transform: 数据转换方法，如果为None则使用默认转换

    返回:
        torch.utils.data.Dataset: MNIST数据集对象
    """
    if transform is None:
        # 使用标准的MNIST数据转换，与test_client.py中的转换保持一致
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # 返回PyTorch格式的MNIST数据集
    return datasets.MNIST(root='./data', train=train, download=True, transform=transform)


# 创建客户端数据加载器
def create_client_dataloaders(train_dataset, num_clients, batch_size, shuffle=True, verbose=True, num_workers=0, pin_memory=True):
    """
    为联邦学习创建客户端数据加载器
    
    Args:
        train_dataset: 完整的训练数据集
        num_clients: 客户端数量
        batch_size: 每个批次的样本数
        shuffle: 是否打乱数据
        verbose: 是否打印详细信息
        num_workers: 数据加载的线程数，默认为4
        pin_memory: 是否将数据固定在内存中以加速GPU传输，默认为True
        
    Returns:
        List[DataLoader]: 每个客户端的数据加载器列表
    """
    client_dataloaders = []
    total_train_size = len(train_dataset)
    client_train_size = total_train_size // num_clients
    
    if verbose:
        info(f"\n切分{num_clients}份数据集...")
        
    # 为每个客户端创建数据加载器
    for i in range(num_clients):
        # 为每个客户端随机选择一部分数据
        client_indices = torch.randperm(total_train_size)[:client_train_size]
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
        client_loader = torch.utils.data.DataLoader(
            client_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
        client_dataloaders.append(client_loader)
        
        if verbose:
            info(f" {i+1} 数据准备完成，样本数: {len(client_dataset)}")
    
    return client_dataloaders


if __name__ == '__main__':
    pass