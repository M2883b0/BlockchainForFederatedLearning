# -*- coding:utf-8 -*-
# @FileName :client.py
# @Time :2025/10/9 14:32
# @Author :M2883b0
import torch
from torch import nn
from torch.utils.data import DataLoader
import threading

from .FederatedLearning import Validator
from .FederatedLearning.aggregator import Aggregator
from .FederatedLearning.learner import FederatedLearner, CNNModel

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

## # 初始化目录
import os
if not os.path.exists("./global_model"):
    os.mkdir("./global_model")
if not os.path.exists("./client_gradients"):
    os.mkdir("./client_gradients")


class Client:
    def __init__(self, epochs: int, client_name: str, data_loader: DataLoader, ModelClass, main_dict: dict, base_path: str = "."):
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
        self.global_model = ModelClass()
        self.base_path = base_path

    def get_role(self):
        """
        从队列中获取当前轮次的角色
        
        Returns:
            str: 角色名称
        """
        self.role = self.main_dict["role"][-1][self.name]
        print(f"[{self.name}] 当前轮次角色: {self.role}")
        return self.role

    def get_role_nums(self):
        """
        获取当前轮次的角色数量

        Returns:
            int: 角色数量
        """
        role_dict = self.main_dict["role"][self.round]
        roles_li = role_dict.items()
        return roles_li.count("learner"), roles_li.count("validator")

    def load_gradient(self, gradient_path):
        return torch.load(gradient_path)

    def save_gradient(self, gradient, base_path):
        path = f"{base_path}/{self.name}_{self.round}_gradient.pt"
        torch.save(gradient, path)
        return path

    def get_global_model(self):
        """
        获取全局模型

        Returns:
            torch.nn.Module: 全局模型实例
        """
        global_model_path = self.main_dict["global_model"][-1]
        global_model_weight = torch.load(global_model_path)
        self.global_model.load_state_dict(global_model_weight)
        return self.global_model

    def save_global_model(self, base_path):
        path = f"{base_path}/{self.name}_{self.round}_global_model.pt"
        torch.save(self.global_model.state_dict(), path)
        return path

    def sign(self, message):
        return self.name

    def verify(self, message, signature):
        return True

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
            self._run_as_learner()
            # 模拟将梯度发送给聚合者（在实际应用中可能通过网络或区块链传输）
            # self._send_gradients(gradients)
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
        self.get_global_model()
        # 创建聚合器实例
        aggregator = Aggregator(self.global_model)
        
        print(f"[{self.name}] 聚合者开始收集梯度...")
        # 获取投票和梯度列表，等待所有客户端开始梯度计算, 验证者开始投票
        while True:
            cg_list = self.main_dict["client_gradients"]
            votes_list = self.main_dict["votes"]
            if cg_list and votes_list and len(cg_list)== len(votes_list) == self.round + 1:
                client_gradients = cg_list[self.round]
                votes = votes_list[self.round]
                break
        # 获取训练者数量
        learner_nums ,validator_nums = self.get_role_nums()
        # 等待所有客户端完成梯度计算，验证者完成投票
        while True:
            if len(client_gradients) == learner_nums and len(votes) == validator_nums * learner_nums:
                break
        for client_sign, gradient_path in client_gradients:
            vote_of_client = [i[1] for i in votes if i[0] == client_sign]
            if vote_of_client.count(True) > validator_nums // 2:
                self.main_dict["contribution"][client_sign] = 1
                gradient = self.load_gradient(gradient_path)
                aggregator.collect_gradients(gradient)
        # 执行梯度聚合
        print(f"[{self.name}] 开始聚合梯度...")
        aggregated_gradients = aggregator.aggregate()

        # 更新全局模型
        print(f"[{self.name}] 开始更新全局模型...")
        aggregator.update_global_model(aggregated_gradients)

        # 更新本地保存的全局模型
        self.global_model = aggregator.global_model
        path = self.save_global_model(self.base_path + "/global_model")
        self.main_dict["global_model"].append(path)
        print(f"[{self.name}] 全局模型更新完成！")


    def _run_as_learner(self):
        """
        以学习者角色运行
        
        Returns:
            Dict[str, torch.Tensor]: 训练后的梯度
        """
        # 获取全局模型
        self.get_global_model()
        
        # 创建学习者实例
        learner = FederatedLearner(self.global_model, self.data_loader, epochs=self.epochs)
        
        # 执行本地训练
        print(f"[{self.name}] 学习者开始本地训练...")
        train_results = learner.train()
        
        print(f"[{self.name}] 训练完成！损失: {train_results['loss']:.4f}, 准确率: {train_results['accuracy']:.2f}%")
        
        # 导出梯度
        gradient = learner.export_gradients()
        gradient_path = self.save_gradient(gradient, self.base_path + "/client_gradients")
        if not self.main_dict["client_gradients"]:
            self.main_dict["client_gradients"] = [(self.sign(gradient), gradient_path)]
        elif len(self.main_dict["client_gradients"]) == self.round:
            self.main_dict["client_gradients"].append([(self.sign(gradient), gradient_path)])
        elif len(self.main_dict["client_gradients"]) == self.round + 1:
            self.main_dict["client_gradients"][self.round].append((self.sign(gradient), gradient_path))
        else:
            print(f"[{self.name}] 梯度列表长度错误！")
            raise ValueError("无效的梯度列表长度")
        print(f"[{self.name}] 梯度导出完成！")

    def _run_as_validator(self):
        """
        以验证者角色运行
        """
        # 获取全局模型
        self.get_global_model()
        
        print(f"[{self.name}] 验证者开始验证模型...")
        
        # 创建验证器实例
        # 评估当前全局模型
        validator = Validator()
        validator.load_global_model(self.global_model)
        validator.set_test_loader(self.data_loader)
        base_performance = validator.calculate_base_performance()
        # 等待训练者开始训练
        while True:
            cg_list = self.main_dict["client_gradients"]
            if cg_list and len(self.main_dict["client_gradients"]) == self.round + 1:
                client_gradients = cg_list[self.round]
                break
        count = 0
        learner_nums, validator_nums = self.get_role_nums()
        while count < learner_nums:
            if len(client_gradients) > count:
                gradient_sign, gradient_path = client_gradients[count]
                gradient = self.load_gradient(gradient_path)
                valid_result = validator.apply_gradients_and_validate_performance(gradient)
                if not self.verify(gradient_sign, gradient):
                    result = (gradient_sign, self.sign({gradient_sign: False}), False)
                elif valid_result.get("is_acceptable"):
                    result = (gradient_sign, self.sign({gradient_sign: True}), True)
                else:
                    print("精度下降过多: ", valid_result)
                    result = (gradient_sign, self.sign({gradient_sign: False}), False)
                if not self.main_dict["votes"]:
                    self.main_dict["votes"].append([result])
                elif len(self.main_dict["votes"]) == self.round:
                    self.main_dict["votes"].append([result])
                elif len(self.main_dict["votes"]) == self.round + 1:
                    self.main_dict["votes"][self.round].append(result)
                else:
                    print("投票列表长度错误")
                    raise ValueError("无效的投票列表")
                count += 1
                print(f"[{self.name}] 验证者对客户端{gradient_sign}的签名进行验证结果: {result}")
