# -*- coding:utf-8 -*-
# @FileName :learner.py
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
from DGS_BCFL.src.utils.logger import setup_logger, info, debug, warning, error


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
        if device is None or device == "":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # info(f"FederatedLearner使用设备: {self.device}")
        #
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
            count = 0
            for data, target in self.data_loader:
                count += 1
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

            # 记录每轮训练结果
            info(
                f'客户端训练轮次 {epoch + 1}/{self.epochs}: 损失={epoch_loss / len(self.data_loader):.4f}, 准确率={100 * epoch_correct / epoch_total:.2f}%')

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


if __name__ == '__main__':
    # 提示用户使用test_FL.py进行测试
    info("请使用 test_FL.py 文件来运行联邦学习系统测试。")
    info("测试脚本包含完整的多客户端联邦学习训练和聚合过程。")
