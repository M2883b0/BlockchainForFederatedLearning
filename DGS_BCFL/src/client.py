# -*- coding:utf-8 -*-
# @FileName :client.py
# @Time :2025/10/9 14:32
# @Author :M2883b0
import torch
from torch.utils.data import DataLoader
import threading

from .FederatedLearning.aggregator import Aggregator
from .FederatedLearning.learner import FederatedLearner


# main_dict = {
#     "role": [
#         {"client_1": "leaner"}
#     ],
#     "global_model": [
#         ""
#     ],
#     "client_gradients": [
#
#     ],
#     "votes": [
#         ["client_1", "client_2"]
#       ],
#     "contribution": {
#         "client_1": 0.5
#     }
# }

class Client:
    def __init__(self, epochs: int, client_name: str, data_loader: DataLoader, main_dict: dict):
        """
        初始化客户端
        
        Args:
            epochs: 训练轮数
            client_name: 客户端名称
            data_loader: 数据加载器
            role_queue: 角色队列，用于存储每轮的角色信息
        """
        self.role = ""
        self.epochs = epochs
        self.round = 0
        self.name = client_name
        self.data_loader = data_loader
        self.main_dict = main_dict
        self.global_model = None  # 默认使用CNN模型作为全局模型

    def get_global_model(self):
        """
        获取全局模型
        
        Returns:
            torch.nn.Module: 全局模型实例
        """
        self.global_model = self.main_dict["global_model"][-1]
        return self.global_model

    def get_role(self):
        """
        从队列中获取当前轮次的角色
        
        Returns:
            str: 角色名称
        """
        self.role = self.main_dict["role"][-1][self.name]
        print(f"[{self.name}] 当前轮次角色: {self.role}")
        return self.role

    def run(self):
        """
        运行客户端，根据每轮的角色执行相应的任务
        """
        # 获取当前轮次的角色
        current_role = self.get_role()
        if current_role == "aggregator":
            # 聚合者角色
            self._run_as_aggregator()
        elif current_role == "learner":
            # 学习者角色
            gradients = self._run_as_learner()
            # 模拟将梯度发送给聚合者（在实际应用中可能通过网络或区块链传输）
            self._send_gradients(gradients)
        elif current_role == "validator":
            # 验证者角色
            self._run_as_validator()
        else:
            raise ValueError(f"无效的客户端角色: {current_role}")

        print(f"[{self.name}] {current_role}结束第{self.round + 1}轮任务")
        self.round += 1
        print("=" * 70)

    def _run_as_aggregator(self):
        """
        以聚合者角色运行
        """
        # 获取全局模型
        global_model = self.get_global_model()
        # 创建聚合器实例
        aggregator = Aggregator(global_model)
        
        print(f"[{self.name}] 聚合者开始收集梯度...")
        # 等待所有客户端开始梯度计算, 验证者开始投票
        while True:
            cg_list = self.main_dict["client_gradients"]
            votes_list = self.main_dict["votes"]
            if cg_list and votes_list and len(cg_list)== len(votes_list) == self.round + 1:
                client_gradients = cg_list[self.round]
                votes = votes_list[self.round]
                break
        # 获取训练者数量
        role_dict = self.main_dict["role"][self.round]
        roles_li = role_dict.items()
        learner_nums = roles_li.count("learner")
        validator_nums = roles_li.count("validator")
        # 等待所有客户端完成梯度计算，验证者完成投票
        while True:
            if len(client_gradients) == learner_nums and len(votes) == validator_nums:
                break
        for gradients in self.collected_gradients:
            aggregator.collect_gradients(gradients)

        # 执行梯度聚合
        print(f"[{self.name}] 开始聚合梯度...")
        aggregated_gradients = aggregator.aggregate()

        # 更新全局模型
        print(f"[{self.name}] 开始更新全局模型...")
        aggregator.update_global_model(aggregated_gradients)

        # 更新本地保存的全局模型
        self.global_model = aggregator.global_model

        print(f"[{self.name}] 全局模型更新完成！")

    def _run_as_learner(self):
        """
        以学习者角色运行
        
        Returns:
            Dict[str, torch.Tensor]: 训练后的梯度
        """
        # 获取全局模型
        global_model = self.get_global_model()
        
        # 创建学习者实例
        learner = FederatedLearner(global_model, self.data_loader, epochs=1)
        
        # 执行本地训练
        print(f"[{self.name}] 学习者开始本地训练...")
        train_results = learner.train()
        
        print(f"[{self.name}] 训练完成！损失: {train_results['loss']:.4f}, 准确率: {train_results['accuracy']:.2f}%")
        
        # 导出梯度
        gradients = learner.export_gradients()
        print(f"[{self.name}] 梯度导出完成！")
        
        return gradients

    def _run_as_validator(self):
        """
        以验证者角色运行
        """
        # 获取全局模型
        global_model = self.get_global_model()
        
        print(f"[{self.name}] 验证者开始验证模型...")
        
        # 创建验证器实例
        # 注意：在当前版本的aggregator中，验证器功能已被移除
        # 这里简单模拟验证过程
        
        # 评估当前全局模型
        validator = FederatedLearner(global_model, self.data_loader, epochs=0)
        if self.data_loader is not None:
            evaluation_results = validator.evaluate(self.data_loader)
            print(f"[{self.name}] 模型验证结果: 损失={evaluation_results['loss']:.4f}, 准确率={evaluation_results['accuracy']:.2f}%")
        else:
            print(f"[{self.name}] 没有验证数据，无法执行验证！")

    def _send_gradients(self, gradients):
        """
        模拟发送梯度给聚合者
        
        Args:
            gradients: 要发送的梯度
        """
        # 在实际应用中，这里应该通过网络或区块链将梯度发送给聚合者
        # 这里简单保存梯度，供后续聚合使用
        print(f"[{self.name}] 正在发送梯度...")
        if gradients:
            self.collected_gradients.append(gradients)
            print(f"[{self.name}] 梯度发送完成！")
        else:
            print(f"[{self.name}] 没有有效梯度可发送！")
