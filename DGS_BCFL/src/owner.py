# -*- coding: utf-8 -*-
# @Time    : 2025/10/12 11:36
# @Author  : M2883b0
# @File    : owner.py
import copy
import time
from threading import Lock
import torch
from sympy.physics.units import action
from triton.profiler import activate, deactivate

from utils.logger import info


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
            "contribution": {},
            "global_accuracy_history": [],
            "contribution_history": [],
            "active_clients": [],
            "suspicious_clients": [],
            "deactivate_clients": []
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
        self.lock = Lock()

    def get_main_dict(self):
        """
        获取主字典
        """
        return self.main_dict

    def get_lock(self):
        """
        获取锁
        """
        return self.lock

    def join(self, client_name: str):
        """
        加入客户端
        """
        with self.lock:
            self.main_dict["contribution"][client_name] = 0
            self.main_dict["active_clients"].append(client_name)

    def distribute_incentives(self):
        """
        分发奖励
        """
        with self.lock:
            # 验证者奖励计算
            num_validator = num_learner = 0
            for client_name, role in self.main_dict["role"][self.round].items():
                if role == "validator":
                    num_validator += 1
                elif role == "learner":
                    num_learner += 1
            vote_len = len(self.main_dict["votes"][self.round])
            S_v = 0
            r_v = 0.1
            for _, _, data_len, _ in self.main_dict["client_gradients"][self.round]:
                S_v += data_len
            S_v = S_v / num_learner
            for i, (_, validator, _, _, _)in enumerate(self.main_dict["votes"][self.round]):
                self.main_dict["contribution"][validator] += (vote_len -i) /vote_len * S_v * r_v

            # 训练者奖励计算
            r_l = 1
            tao = 10
            PT_dict = {client: 0.0 for client, role in self.main_dict["role"][self.round].items() if role == "learner"}
            sum_validator = sum([self.main_dict["contribution"][client] for client, role in self.main_dict["role"][self.round].items() if role == "validator"])
            for learner, validator, flag, data_len,  epochs in self.main_dict["votes"][self.round]:
                PT_dict[learner] += flag * self.main_dict["contribution"][validator]
            for key in PT_dict:
                PT_dict[key] = PT_dict[key] / sum_validator
            for learner_sign, _, data_len, epochs in self.main_dict["client_gradients"][self.round]:
                # 防止过度训练
                if epochs > tao:
                    epochs = -(epochs - tao)
                # 只对接受的梯度计算贡献
                if PT_dict[learner_sign] >= 0.5:
                    self.main_dict["contribution"][learner_sign] += data_len * epochs * r_l
                else:
                    self.main_dict["contribution"][learner_sign] += 0


    def assign_roles(self):
        """
        重新分配角色， 剔除恶意客户端
        """
        contribution = list(self.main_dict["contribution"].items())
        contribution = sorted(contribution, key=lambda x: x[1], reverse=True)
        old_suspicious_clients = self.main_dict["suspicious_clients"]

        activate_client = self.main_dict["active_clients"]
        deactivate_client = self.main_dict["deactivate_clients"]
        if contribution[-1][0] == old_suspicious_clients:
            activate_client.remove(old_suspicious_clients)
            deactivate_client.append(old_suspicious_clients)
        n = len(activate_client)
        aggregators_num = 1
        validators_num = (n - 1) // 3
        learners_num = n - aggregators_num - validators_num
        role_dict = {}
        for client_name, contribution in contribution:
            if client_name in activate_client:
                if aggregators_num > 0:
                    role_dict[client_name] = "aggregator"
                    aggregators_num -= 1
                elif validators_num > 0:
                    role_dict[client_name] = "validator"
                    validators_num -= 1
                else:
                    role_dict[client_name] = "learner"
                    learners_num -= 1
            else:
                role_dict[client_name] = "deactivate_client"
        with self.lock:
            self.main_dict["role"].append(copy.deepcopy(role_dict))

    def run(self):
        """
        运行
        """
        if self.round == 0:
            self.assign_roles()
            info(f"Owner开始初始化 角色 {self.main_dict['role']}")
            # info(f"初始化完成 {self.main_dict}")
        if self.round == 0 and self.main_dict["contribution_history"] == []:
            with self.lock:
                self.main_dict["contribution_history"].append(copy.deepcopy(self.main_dict["contribution"]))
            info(f"开始初始化contribution_history : {self.main_dict['contribution_history']}" )
        while True:
            if self.round + 2 == len(self.main_dict["global_model"]):
                break
            time.sleep(1)
            # info(f"Owner time To print main_dict {self.main_dict}")
        # 分配激励
        self.distribute_incentives()
        info(f'contributions is :{self.main_dict["contribution"]}')
        self.round += 1
        # 每隔一定轮数，重新分配角色
        if self.round % self.rotation_cycle == 0:
            self.assign_roles()
        else:
            with self.lock:
                self.main_dict["role"].append(copy.deepcopy(self.main_dict["role"][-1]))
        # info(f"追加记录role : {self.main_dict['role']}")
        with self.lock:
            self.main_dict["contribution_history"].append(copy.deepcopy(self.main_dict["contribution"]))
            # info(f"追加记录contribution_history : {self.main_dict['contribution_history']}")
        # info(f"Owner运行完成 {self.main_dict}")
        info(f'contributions is :{self.main_dict["contribution"]}')

class MyOwner(Owner):
    def __init__(self, rotation_cycle: int, model_class):
        super().__init__(rotation_cycle, model_class)

    def distribute_incentives(self):
        """
        分发奖励, 并找出可疑客户端
        """
        suspicious_clients = []
        with self.lock:
            # 验证者奖励计算
            num_validator = num_learner = 0
            for client_name, role in self.main_dict["role"][self.round].items():
                if role == "validator":
                    num_validator += 1
                elif role == "learner":
                    num_learner += 1
            vote_len = len(self.main_dict["votes"][self.round])
            S_v = 0
            r_v = 0.1
            for _, _, data_len, _ in self.main_dict["client_gradients"][self.round]:
                S_v += data_len
            S_v = S_v / num_learner
            for i, (_, validator, _, _, _)in enumerate(self.main_dict["votes"][self.round]):
                self.main_dict["contribution"][validator] += (vote_len -i) /vote_len * S_v * r_v

            # 训练者奖励计算
            r_l = 1
            tao = 10
            PT_dict = {client: 0.0 for client, role in self.main_dict["role"][self.round].items() if role == "learner"}
            sum_validator = sum([self.main_dict["contribution"][client] for client, role in self.main_dict["role"][self.round].items() if role == "validator"])
            for learner, validator, flag, data_len,  epochs in self.main_dict["votes"][self.round]:
                PT_dict[learner] += flag * self.main_dict["contribution"][validator]
            for key in PT_dict:
                PT_dict[key] = PT_dict[key] / sum_validator
                if PT_dict[key] < 0.5:
                    suspicious_clients.append(key)
            for learner_sign, _, data_len, epochs in self.main_dict["client_gradients"][self.round]:
                # 防止过度训练
                if epochs > tao:
                    epochs = -(epochs - tao)
                # 只对接受的梯度计算贡献
                if PT_dict[learner_sign] >= 0.5:
                    self.main_dict["contribution"][learner_sign] += data_len * epochs * r_l
                else:
                    self.main_dict["contribution"][learner_sign] += 0

    def assign_roles(self):
        """
        重新分配角色
        """
        contribution = list(self.main_dict["contribution"].items())
        contribution = sorted(contribution, key=lambda x: x[1], reverse=True)
        activate_client = self.main_dict["active_clients"]
        deactivate_client = self.main_dict["deactivate_clients"]
        n = len(activate_client)
        aggregators_num = 1
        validators_num = (n - 1) // 3
        regulators_num = (validators_num) // 3
        learners_num = n - aggregators_num - validators_num - regulators_num
        role_dict = {}
        for client_name, contribution in contribution:
            if client_name in activate_client:
                if aggregators_num > 0:
                    role_dict[client_name] = "aggregator"
                    aggregators_num -= 1
                elif  regulators_num > 0:
                    role_dict[client_name] = "regulators"
                    regulators_num -= 1
                elif validators_num > 0:
                    role_dict[client_name] = "validator"
                    validators_num -= 1
                else:
                    role_dict[client_name] = "learner"
                    learners_num -= 1
            else:
                role_dict[client_name] = "deactivate_client"
        with self.lock:
            self.main_dict["role"].append(copy.deepcopy(role_dict))


if __name__ == "__main__":
    pass
