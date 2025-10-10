# -*- coding:utf-8 -*-
# @FileName :test_client.py
# @Time :2025/10/9 20:25
# @Author :M2883b0

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from queue import Queue

from DGS_BCFL.src.FederatedLearning.learner import FederatedLearner, CNNModel
from DGS_BCFL.src.client import Client


def prepare_test_data():
    """
    准备测试用的MNIST数据集
    
    Returns:
        DataLoader: 测试数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader


def run_client_test():
    """
    运行客户端测试
    """
    print("开始运行客户端测试...")
    
    # 准备测试数据
    test_loader = prepare_test_data()
    
    # 创建角色队列，定义每轮的角色
    role_queue = Queue()
    # 添加三轮的角色：learner -> aggregator -> validator
    role_queue.put("learner")
    role_queue.put("aggregator")
    role_queue.put("validator")
    
    # 创建客户端实例
    client = Client(epochs=3, client_name="TestClient", data_loader=test_loader, role_queue=role_queue)
    
    # 运行客户端
    client.run()
    
    print("客户端测试完成！")


if __name__ == '__main__':
    """
    客户端测试入口
    """
    run_client_test()
