# -*- coding:utf-8 -*-
# @FileName :aggregator.py
# @Time :2025/10/7 21:22
# @Author :M2883b0

"""
- 区块链联邦学习系统 -
  联邦学习聚合器

  本模块实现了联邦学习中的梯度聚合功能，包括：
  1. 客户端梯度的收集与聚合
  2. 梯度有效性验证
  3. 全局模型更新
  4. 聚合策略管理
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time


class Aggregator:
    """
    联邦学习聚合器类，负责收集、验证和聚合客户端的梯度更新
    """
    
    def __init__(self, global_model: torch.nn.Module, learning_rate: float = 0.01,
                 gradient_threshold: float = None, max_gradients: int = None,
                 device: str = None, test_loader: Optional[torch.utils.data.DataLoader] = None):
        """
        初始化联邦学习聚合器
        
        Args:
            global_model: 全局模型实例
            learning_rate: 学习率，默认0.01
            gradient_threshold: 梯度阈值，用于异常检测，默认为None（不启用）
            max_gradients: 最大允许的梯度数量，用于限制参与聚合的客户端数量
            device: 计算设备，默认自动判断(优先GPU)
            test_loader: 测试数据加载器，用于模型评估
        """
        # 自动判断设备，如果未指定则优先使用GPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Aggregator使用设备: {self.device}")
        
        self.global_model = global_model.to(self.device)
        self.learning_rate = learning_rate
        self.gradient_threshold = gradient_threshold
        self.max_gradients = max_gradients
        self.collected_gradients = []
        self.collected_weights = []
        self.test_loader = test_loader
    
    def collect_gradients(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> bool:
        """
        收集客户端提交的梯度
        
        Args:
            gradients: 客户端计算的梯度字典
            weight: 客户端权重，用于加权聚合，默认1.0
            
        Returns:
            bool: 梯度收集是否成功
        """
        # 简单验证梯度有效性（检查是否为空）
        if not gradients:
            print("警告：接收到空梯度，已拒绝")
            return False
        
        # 检查梯度数量限制
        if self.max_gradients is not None and len(self.collected_gradients) >= self.max_gradients:
            print(f"警告：已达到最大梯度收集数量 {self.max_gradients}")
            return False
        
        # 可选：检查梯度范数（如果设置了阈值）
        if self.gradient_threshold is not None:
            try:
                for param_name, grad in gradients.items():
                    grad_norm = torch.norm(grad).item()
                    if grad_norm > self.gradient_threshold:
                        print(f"警告：梯度 '{param_name}' 的范数 {grad_norm:.2f} 超过阈值 {self.gradient_threshold}")
                        return False
            except Exception as e:
                print(f"警告：检查梯度范数时出错: {e}")
        
        # 收集梯度和权重
        self.collected_gradients.append({k: v.to(self.device) for k, v in gradients.items()})
        self.collected_weights.append(weight)
        
        return True
    
    def aggregate(self, strategy: str = 'average', reset_after: bool = True) -> Dict[str, torch.Tensor]:
        """
        聚合收集到的梯度
        
        Args:
            strategy: 聚合策略，可选 'average'(平均) 或 'weighted'(加权平均)
            reset_after: 聚合后是否重置收集的梯度
            
        Returns:
            Dict[str, torch.Tensor]: 聚合后的梯度
        """
        if not self.collected_gradients:
            print("警告：没有可聚合的梯度")
            return {}
        
        print(f"\n开始聚合{len(self.collected_gradients)}个客户端的梯度...")
        aggregation_start_time = time.time()
        
        # 根据策略选择是否使用权重
        if strategy.lower() == 'weighted' and len(self.collected_weights) == len(self.collected_gradients):
            weights = self.collected_weights
        else:
            weights = None
        
        # 执行梯度聚合
        aggregated_gradients = self._aggregate_gradients(self.collected_gradients, weights)
        
        aggregation_time = time.time() - aggregation_start_time
        print(f"梯度聚合完成! 耗时: {aggregation_time:.2f}秒")
        print(f"聚合后的梯度参数数量: {len(aggregated_gradients)}")
        
        # 重置收集的梯度（如果需要）
        if reset_after:
            self.reset_collected_gradients()
        
        return aggregated_gradients
    
    def _aggregate_gradients(self, gradients_list: List[Dict[str, torch.Tensor]], 
                            weights: List[float] = None) -> Dict[str, torch.Tensor]:
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
    
    def update_global_model(self, gradients: Dict[str, torch.Tensor], 
                           learning_rate: float = None) -> None:
        """
        使用聚合后的梯度更新全局模型
        
        Args:
            gradients: 聚合后的梯度
            learning_rate: 学习率，如果为None则使用初始化时的学习率
        """
        if not gradients:
            print("警告：没有可用的梯度来更新模型")
            return
        
        # 使用指定的学习率或默认学习率
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        print("更新全局模型...")
        update_start_time = time.time()
        
        # 应用梯度更新模型参数
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in gradients and param.requires_grad:
                    param.data += lr * gradients[name]
        
        update_time = time.time() - update_start_time
        print(f"全局模型更新完成! 耗时: {update_time:.2f}秒")
    
    def reset_collected_gradients(self) -> None:
        """
        重置收集的梯度和权重
        """
        self.collected_gradients = []
        self.collected_weights = []
    
    def get_collected_count(self) -> int:
        """
        获取当前收集的梯度数量
        
        Returns:
            int: 已收集的梯度数量
        """
        return len(self.collected_gradients)
    
    def evaluate_model(self, test_loader: torch.utils.data.DataLoader, 
                      criterion: torch.nn.Module = None) -> Dict[str, float]:
        """
        在测试数据集上评估全局模型性能
        
        Args:
            test_loader: 测试数据加载器
            criterion: 损失函数，如果为None则使用CrossEntropyLoss
            
        Returns:
            Dict: 包含测试损失和准确率的字典
        """
        # 使用指定的损失函数或默认的交叉熵损失函数
        if criterion is None:
            eval_criterion = torch.nn.CrossEntropyLoss()
        else:
            eval_criterion = criterion
        
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += eval_criterion(output, target).item()
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
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """
        获取收集的梯度统计信息
        
        Returns:
            Dict: 包含梯度统计信息的字典
        """
        if not self.collected_gradients:
            return {}
        
        stats = {
            'total_gradients': len(self.collected_gradients),
            'gradient_norms': {},
            'mean_gradient': {},
            'std_gradient': {}
        }
        
        # 获取第一个客户端的梯度键
        first_client_grads = self.collected_gradients[0]
        
        # 计算每个参数的统计信息
        for param_name in first_client_grads.keys():
            # 收集所有客户端的该参数梯度
            param_grads = [client_grads[param_name] for client_grads in self.collected_gradients]
            
            # 计算梯度范数
            norms = [torch.norm(grad).item() for grad in param_grads]
            stats['gradient_norms'][param_name] = {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'min': np.min(norms),
                'max': np.max(norms)
            }
            
            # 计算梯度的均值和标准差
            flattened_grads = [grad.cpu().numpy().flatten() for grad in param_grads]
            all_grads = np.concatenate(flattened_grads)
            stats['mean_gradient'][param_name] = np.mean(all_grads)
            stats['std_gradient'][param_name] = np.std(all_grads)
        
        return stats


if __name__ == '__main__':
    """
    聚合器类的简单示例
    """
    import torchvision
    import torchvision.transforms as transforms
    from src.FederatedLearning.learner import CNNModel, FederatedLearner
    
    # 准备数据
    def prepare_data():
        """准备MNIST数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                  download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                 download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                  shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                 shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    
    # 创建简单的测试客户端
    def create_test_clients(global_model, train_loader, num_clients=2):
        """创建测试客户端"""
        clients = []
        # 简单地将训练数据分成num_clients份
        dataset_size = len(train_loader.dataset)
        client_size = dataset_size // num_clients
        
        for i in range(num_clients):
            # 为每个客户端创建数据加载器
            client_dataset = torch.utils.data.Subset(
                train_loader.dataset,
                range(i * client_size, (i + 1) * client_size)
            )
            client_loader = torch.utils.data.DataLoader(client_dataset, 
                                                      batch_size=train_loader.batch_size,
                                                      shuffle=True, num_workers=0)
            
            # 创建客户端实例
            client = FederatedLearner(global_model, client_loader, learning_rate=0.01, epochs=1)
            clients.append(client)
        
        return clients
    
    # 运行测试
    def run_test():
        """运行聚合器测试"""
        # 准备数据
        train_loader, test_loader = prepare_data()
        
        # 创建全局模型
        global_model = CNNModel()
        
        # 创建聚合器
        aggregator = Aggregator(
            global_model=global_model,
            learning_rate=0.01,
            gradient_threshold=20.0,
            max_gradients=10,
            test_loader=test_loader
        )
        
        # 创建客户端
        clients = create_test_clients(global_model, train_loader, num_clients=3)
        
        # 模拟一轮联邦学习
        print("\n===== 开始模拟联邦学习过程 =====")
        
        # 客户端训练并提交梯度
        for i, client in enumerate(clients):
            print(f"\n客户端 {i+1} 开始训练...")
            # 客户端加载全局模型
            client.load_global_model(global_model)
            # 客户端训练
            train_result = client.train()
            print(f"客户端 {i+1} 训练完成: 损失={train_result['loss']:.4f}, 准确率={train_result['accuracy']:.2f}%")
            # 客户端导出梯度
            gradients = client.export_gradients()
            # 聚合器收集梯度
            success = aggregator.collect_gradients(gradients, weight=1.0)
            print(f"客户端 {i+1} 梯度收集{'成功' if success else '失败'}")
        
        # 聚合梯度
        print("\n开始聚合梯度...")
        aggregated_gradients = aggregator.aggregate(strategy='average')
        
        # 更新全局模型
        print("\n更新全局模型...")
        aggregator.update_global_model(aggregated_gradients)
        
        # 评估更新后的模型
        print("\n评估更新后的全局模型...")
        performance = aggregator.evaluate_model(test_loader)
        print(f"全局模型性能: 损失={performance['loss']:.4f}, 准确率={performance['accuracy']:.2f}%")
        
    # 运行测试
    if __name__ == '__main__':
        run_test()
