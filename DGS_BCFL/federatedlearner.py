# -*- coding:utf-8 -*-
# @FileName :federatedlearner.py
# @Time :2025/10/7 21:21
# @Author :M2883b0

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from typing import Dict, List, Any, Tuple
import torchvision
import torchvision.transforms as transforms
import time

# 导入数据处理模块
import data_split


# 全局模型定义
class SimpleModel(nn.Module):
    """简单的全连接神经网络模型"""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平输入图像
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel(nn.Module):
    """简单的卷积神经网络模型"""

    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepCNNModel(nn.Module):
    """更深的卷积神经网络模型"""

    def __init__(self):
        super(DeepCNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc_layers(x)
        return x


class MLPModel(nn.Module):
    """多层感知器模型"""

    def __init__(self):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layers(x)
        return x


# 模型选择字典
MODEL_CHOICES = {
    'simple': SimpleModel,
    'cnn': CNNModel,
    'deep_cnn': DeepCNNModel,
    'mlp': MLPModel
}


class FederatedLearner:
    """
    联邦学习客户端类，负责本地模型训练、梯度计算和模型更新等功能
    """

    def __init__(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                 learning_rate: float = 0.01, epochs: int = 5, device: str = None):
        """
        初始化联邦学习客户端
        
        Args:
            model: 深度学习模型实例
            data_loader: 数据加载器，用于加载本地训练数据
            learning_rate: 学习率，默认0.01
            epochs: 本地训练轮数，默认5
            device: 训练设备，默认自动判断(优先GPU)
        """
        # 自动判断设备，如果未指定则优先使用GPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"FederatedLearner使用设备: {self.device}")
            
        self.model = copy.deepcopy(model).to(self.device)
        self.global_model = copy.deepcopy(model).to(self.device)
        self.data_loader = data_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self) -> Dict[str, Any]:
        """
        在本地数据集上训练模型
        
        Returns:
            Dict: 包含训练结果的字典，包括损失、准确率等指标
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 梯度清零
                self.optimizer.zero_grad()

                # 前向传播
                output = self.model(data)
                loss = self.criterion(output, target)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                # 统计损失和准确率
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()

            # 累计总损失和准确率
            total_loss += epoch_loss / len(self.data_loader)
            correct += epoch_correct
            total += epoch_total

            # 打印每轮训练结果
            print(f'客户端训练轮次 {epoch + 1}/{self.epochs}: 损失={epoch_loss / len(self.data_loader):.4f}, ' f'准确率={100 * epoch_correct / epoch_total:.2f}%')

        # 计算平均损失和准确率
        avg_loss = total_loss / self.epochs
        accuracy = 100 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epochs': self.epochs,
            'samples': total
        }

    def load_global_model(self, global_model: nn.Module) -> None:
        """
        加载全局模型参数到本地
        
        Args:
            global_model: 从服务器获取的全局模型
        """
        self.global_model = copy.deepcopy(global_model).to(self.device)
        self.model.load_state_dict(self.global_model.state_dict())
        # 重新初始化优化器以适应新的模型参数
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def export_gradients(self) -> Dict[str, torch.Tensor]:
        """
        计算并导出本地模型与全局模型之间的梯度差异
        
        Returns:
            Dict: 包含模型各层梯度的字典
        """
        gradients = {}

        # 计算每个参数的梯度（当前模型参数 - 全局模型参数）
        for (name, global_param), (_, local_param) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()):
            if local_param.requires_grad:
                gradients[name] = local_param.data - global_param.data

        return gradients

    def export_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        导出本地训练后的模型参数
        
        Returns:
            Dict: 包含模型所有参数的字典
        """
        return copy.deepcopy({name: param.data for name, param in self.model.named_parameters()})

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        在测试数据集上评估模型性能
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            Dict: 包含测试损失和准确率的字典
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total
        }

    def get_model_size(self) -> int:
        """
        获取模型参数量大小（以字节为单位）
        
        Returns:
            int: 模型参数量大小
        """
        size = 0
        for param in self.model.parameters():
            size += param.nelement() * param.element_size()
        return size


def aggregate_gradients(gradients_list: List[Dict[str, torch.Tensor]], weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """
    聚合多个客户端的梯度
    
    Args:
        gradients_list: 客户端梯度列表
        weights: 客户端权重列表，用于加权聚合
    
    Returns:
        Dict: 聚合后的梯度
    """
    if not gradients_list:
        return {}
    
    # 初始化聚合梯度字典
    aggregated_gradients = {}
    
    # 获取第一个客户端的梯度键
    first_client_grads = gradients_list[0]
    
    # 为每个参数计算聚合梯度
    for param_name in first_client_grads.keys():
        # 收集所有客户端的该参数梯度
        param_grads = [client_grads[param_name] for client_grads in gradients_list]
        
        if weights is None:
            # 简单平均
            aggregated_grad = torch.mean(torch.stack(param_grads), dim=0)
        else:
            # 加权平均
            weighted_grads = [weights[i] * grad for i, grad in enumerate(param_grads)]
            aggregated_grad = sum(weighted_grads)
        
        aggregated_gradients[param_name] = aggregated_grad
    
    return aggregated_gradients


def apply_gradients(model: nn.Module, gradients: Dict[str, torch.Tensor], learning_rate: float = 0.01) -> None:
    """
    将梯度应用到模型参数上
    
    Args:
        model: 要更新的模型
        gradients: 聚合后的梯度
        learning_rate: 学习率
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in gradients and param.requires_grad:
                param.data += learning_rate * gradients[name]


if __name__ == '__main__':
    """
    多客户端联邦学习完整示例
    此示例实现了标准的多轮联邦学习训练和聚合过程
    """
    # 设置设备（优先使用GPU，如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"主程序使用设备: {device}")
    
    # 选择模型类型
    model_type = 'cnn'  # 可选: 'simple', 'cnn', 'deep_cnn', 'mlp'
    print(f"选择的模型类型: {model_type}")
    
    # 创建模型实例
    model_class = MODEL_CHOICES.get(model_type, SimpleModel)
    global_model = model_class().to(device)
    
    # 数据预处理和加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    print("\n正在加载MNIST数据集...")
    start_time = time.time()
    
    # 加载MNIST训练集和测试集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 联邦学习参数设置
    num_clients = 3  # 客户端数量
    num_rounds = 3   # 联邦学习轮数
    local_epochs = 2 # 每个客户端本地训练轮数
    learning_rate = 0.01
    batch_size = 32
    
    # 使用data_split模块中的函数为每个客户端分配数据集
    client_dataloaders = data_split.create_client_dataloaders(
        train_dataset=train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        shuffle=True,
        verbose=True
    )
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n=== 开始多客户端联邦学习 ===")
    print(f"模型名称: {model_class.__name__}")
    print(f"客户端数量: {num_clients}, 联邦学习轮数: {num_rounds}")
    print(f"每轮客户端本地训练轮数: {local_epochs}, 学习率: {learning_rate}")
    print(f"模型参数数量: {sum(p.numel() for p in global_model.parameters())}")
    
    # 初始化联邦学习客户端
    clients = []
    for i in range(num_clients):
        # 每个客户端使用相同的全局模型初始化
        client_model = copy.deepcopy(global_model)
        client = FederatedLearner(
            client_model, 
            client_dataloaders[i], 
            learning_rate=learning_rate, 
            epochs=local_epochs
        )
        clients.append(client)
    
    # 存储每轮的性能指标
    round_metrics = []
    
    # 评估初始全局模型性能
    initial_evaluator = FederatedLearner(copy.deepcopy(global_model), None)
    initial_results = initial_evaluator.evaluate(test_loader)
    print(f"\n初始全局模型性能:")
    print(f"损失: {initial_results['loss']:.4f}, 准确率: {initial_results['accuracy']:.2f}%")
    round_metrics.append({
        'round': 0,
        'loss': initial_results['loss'],
        'accuracy': initial_results['accuracy']
    })
    
    # 执行多轮联邦学习
    total_fl_start_time = time.time()
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== 联邦学习第 {round_num}/{num_rounds} 轮 ===")
        
        # 收集所有客户端的梯度
        all_gradients = []
        
        # 每个客户端进行本地训练
        for client_idx, client in enumerate(clients):
            print(f"\n[客户端 {client_idx+1}] 开始本地训练...")
            
            # 确保客户端使用最新的全局模型
            client.load_global_model(global_model)
            
            # 执行本地训练
            client_train_start_time = time.time()
            train_results = client.train()
            client_train_time = time.time() - client_train_start_time
            
            print(f"[客户端 {client_idx+1}] 训练完成!")
            print(f"[客户端 {client_idx+1}] 训练耗时: {client_train_time:.2f}秒")
            print(f"[客户端 {client_idx+1}] 平均训练损失: {train_results['loss']:.4f}")
            print(f"[客户端 {client_idx+1}] 训练准确率: {train_results['accuracy']:.2f}%")
            
            # 导出梯度并收集
            gradients = client.export_gradients()
            all_gradients.append(gradients)
        
        # 服务器聚合梯度
        print(f"\n[服务器] 开始聚合{len(all_gradients)}个客户端的梯度...")
        aggregated_gradients = aggregate_gradients(all_gradients)
        print(f"[服务器] 梯度聚合完成!")
        print(f"[服务器] 聚合后的梯度参数数量: {len(aggregated_gradients)}")
        
        # 应用聚合后的梯度更新全局模型
        print("[服务器] 更新全局模型...")
        apply_gradients(global_model, aggregated_gradients, learning_rate)
        print("[服务器] 全局模型更新完成!")
        
        # 评估当前全局模型性能
        evaluator = FederatedLearner(copy.deepcopy(global_model), None)
        results = evaluator.evaluate(test_loader)
        print(f"\n[服务器] 第 {round_num} 轮全局模型性能:")
        print(f"[服务器] 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")
        
        # 记录本轮性能指标
        round_metrics.append({
            'round': round_num,
            'loss': results['loss'],
            'accuracy': results['accuracy']
        })
    
    total_fl_time = time.time() - total_fl_start_time
    
    print(f"\n=== 多客户端联邦学习训练完成 ===")
    print(f"总训练耗时: {total_fl_time:.2f}秒")
    
    # 打印每轮性能指标
    print("\n各轮性能指标汇总:")
    for metrics in round_metrics:
        print(f"轮次 {metrics['round']}: 损失={metrics['loss']:.4f}, 准确率={metrics['accuracy']:.2f}%")
    
    # 比较初始和最终性能
    initial_accuracy = round_metrics[0]['accuracy']
    final_accuracy = round_metrics[-1]['accuracy']
    accuracy_improvement = final_accuracy - initial_accuracy
    print(f"\n准确率提升: {accuracy_improvement:.2f}%")
    
    print("\n提示: 您可以通过修改以下参数来调整联邦学习过程:\n"
          f"- num_clients: 当前设置为 {num_clients}\n"
          f"- num_rounds: 当前设置为 {num_rounds}\n"
          f"- local_epochs: 当前设置为 {local_epochs}\n"
          f"- model_type: 当前设置为 '{model_type}'")
    
    # print("\n支持的模型类型:\n"
    #       "- 'simple': 简单的全连接神经网络\n"
    #       "- 'cnn': 简单的卷积神经网络\n"
    #       "- 'deep_cnn': 更深的卷积神经网络\n"
    #       "- 'mlp': 多层感知器")
