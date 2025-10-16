# -*- coding: utf-8 -*-
# @Time    : 2025/10/12 11:36
# @Author  : M2883b0
# @File    : owner.py
import time
from threading import Lock
import torch


class Owner:
    """
    Owner类
    """

    def __init__(self, rotation_cycle: int, model_class):
        self.main_dict = {
            "role": [],
            "global_model": [],
            "client_gradients": [],
            "votes": [],
            "contribution": {}
        }
        self.round = 0
        self.rotation_cycle = rotation_cycle
        # 创建目录
        import os
        if not os.path.exists("./global_model"):
            os.mkdir("./global_model")
        if not os.path.exists("./client_gradients"):
            os.mkdir("./client_gradients")
        # 初始化全局模型
        init_global_model = model_class()
        path = "./global_model/init_global_model"
        torch.save(init_global_model.state_dict(), path)
        self.main_dict["global_model"].append(path)

    def get_main_dict(self):
        """
        获取主字典
        """
        return self.main_dict

    def join(self, client_name: str):
        """
        加入客户端
        """
        self.main_dict["contribution"][client_name] = 0

    def distribute_incentives(self):
        """
        分发奖励
        """
        # TODO
        # 训练者奖励计算

        # 验证者奖励计算

    def assign_roles(self):
        """
        重新分配角色
        """
        contribution = list(self.main_dict["contribution"].items())
        contribution.sort(key=lambda x: x[1], reverse=True)
        n = len(contribution)
        aggregators_num = 1
        validators_num = (n - 1) // 3
        learners_num = n - aggregators_num - validators_num
        role_dict = {}
        for client_name, contribution in contribution:
            if aggregators_num > 0:
                role_dict[client_name] = "aggregator"
                aggregators_num -= 1
            elif validators_num > 0:
                role_dict[client_name] = "validator"
                validators_num -= 1
            else:
                role_dict[client_name] = "learner"
                learners_num -= 1
        self.main_dict["role"].append(role_dict)

    def run(self):
        """
        运行
        """
        self.assign_roles()
        while True:
            if self.round + 1 == len(self.main_dict["global_model"]):
                break
            time.sleep(0.5)
        # 分配激励
        self.distribute_incentives()
        # 每隔一定轮数，重新分配角色
        if self.round % self.rotation_cycle == 0:
            self.assign_roles()
        else:
            self.main_dict["role"].append(self.main_dict["role"][-1])


if __name__ == "__main__":
    pass
