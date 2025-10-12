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
    def __init__(self, rotation_cycle:int, model_class):
        self.main_dict = {
            "role": [{}],
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

    def join(self, client_name:str):
        """
        加入客户端
        """

    def distribute_incentives(self):
        """
        分发奖励
        """


    def reassign_roles(self):
        """
        重新分配角色
        """

        if aggregators_num > 0:
            role_dict[client_name] = "aggregator"
            aggregators_num -= 1
        elif validators_num > 0:
            role_dict[client_name] = "validator"
            validators_num -= 1
        else:
            role_dict[client_name] = "learner"
            learners_num -= 1

    def run(self):
        """
        运行
        """
        while True:
            if self.round + 1 == len(self.main_dict["global_model"]):
                break
            time.sleep(0.5)
        # 分配激励
        self.distribute_incentives()
        # 每隔一定轮数，重新分配角色
        if self.round % self.rotation_cycle == 0:
            self.reassign_roles()


if __name__ == "__main__":
    pass
