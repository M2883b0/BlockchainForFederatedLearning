"""
- 区块链联邦学习系统 -
  联邦学习数据提取器
  
  本模块实现了MNIST数据集的获取、分割和持久化存储功能，包括：
  1. 从PyTorch获取MNIST数据集
  2. 数据集的分割，用于联邦学习场景
  3. 数据的保存和加载
  4. 数据集信息显示
"""

import numpy as np
import pickle
import torch
from torchvision import datasets, transforms


def get_mnist():
    """
    使用PyTorch获取MNIST数据集
    
    返回:
        dataset: 包含训练和测试数据的字典
    """
    # 定义数据转换：转换为Tensor并保持原始像素值 (0-255)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)  # 反转归一化，保持0-255范围
    ])

    # 下载并加载数据集
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 准备数据集字典
    dataset = {
        "train_images": train_set.data.numpy(),
        "train_labels": train_set.targets.numpy(),
        "test_images": test_set.data.numpy(),
        "test_labels": test_set.targets.numpy()
    }

    # 添加通道维度 (PyTorch默认没有通道维度)
    dataset["train_images"] = np.expand_dims(dataset["train_images"], axis=-1)
    dataset["test_images"] = np.expand_dims(dataset["test_images"], axis=-1)

    return dataset


def save_data(dataset, name="mnist.d"):
    """
    将数据集以二进制模式保存到文件
    
    参数:
        dataset: 要保存的数据集
        name: 文件名
    """
    with open(name, "wb") as f:
        pickle.dump(dataset, f)


def load_data(name="mnist.d"):
    """
    从二进制文件加载数据集
    
    参数:
        name: 文件名
        
    返回:
        加载的数据集
    """
    with open(name, "rb") as f:
        return pickle.load(f)


def get_dataset_details(dataset):
    """
    显示数据集信息
    
    参数:
        dataset: 要显示信息的数据集
    """
    for k in dataset.keys():
        print(k, dataset[k].shape)
    return


def split_dataset(dataset, split_count):
    """
    将数据集分割成多个联邦数据切片
    
    参数:
        dataset: 要分割的原始数据集
        split_count: 分割数量
        
    返回:
        datasets: 分割后的数据集列表
    """
    datasets = []
    total_samples = len(dataset["train_images"])
    samples_per_split = total_samples // split_count

    for i in range(split_count):
        start_idx = i * samples_per_split
        end_idx = (i + 1) * samples_per_split

        d = {
            "test_images": dataset["test_images"].copy(),
            "test_labels": dataset["test_labels"].copy(),
            "train_images": dataset["train_images"][start_idx:end_idx],
            "train_labels": dataset["train_labels"][start_idx:end_idx]
        }
        datasets.append(d)

    return datasets


if __name__ == '__main__':
    save_data(get_mnist())
    dataset = load_data()
    get_dataset_details(dataset)

    for n, d in enumerate(split_dataset(dataset, 2)):
        save_data(d, "federated_data_" + str(n) + ".d")
        dk = load_data("federated_data_" + str(n) + ".d")
        get_dataset_details(dk)
        print()