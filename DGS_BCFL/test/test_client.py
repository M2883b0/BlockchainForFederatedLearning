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
from DGS_BCFL.src import data_split


if __name__ == '__main__':
    """
    客户端测试入口
    客户端总数 10
    聚合者 1
    验证者 3
    训练者 6
    """
    clients_num = 10
    aggregators_num = 1
    validators_num = 3
    learners_num = 6

    dataset = data_split.get_mnist()
    client_dataloaders = data_split.create_client_dataloaders(dataset["train_images"], 5, 64) * 2
    print(type(client_dataloaders[0]))
    main_dict = {
      "role": [{}],
      "global_model": [],
      "client_gradients":[],
      "votes": [],
      "contribution": {}
    }

    # 初始化客户端以及身份
    clients = []
    role_dict = main_dict["role"][0]
    for i in range(clients_num):
        client_name = f"client_{i+1}"
        client = Client(epochs=5, client_name=client_name, data_loader=client_dataloaders[i], ModelClass=CNNModel, main_dict=main_dict)
        main_dict["contribution"][client_name] = 0
        if aggregators_num > 0:
            role_dict[client_name] = "aggregator"
            aggregators_num -= 1
        elif validators_num > 0:
            role_dict[client_name] = "validator"
            validators_num -= 1
        else:
            role_dict[client_name] = "learner"
            learners_num -= 1
        clients.append(client)

    # 创建目录
    import os
    if not os.path.exists("./global_model"):
        os.mkdir("./global_model")
    if not os.path.exists("./client_gradients"):
        os.mkdir("./client_gradients")
    # 初始化全局模型
    init_global_model = CNNModel()
    path = "./global_model/init_global_model"
    torch.save(init_global_model.state_dict(), path)
    main_dict["global_model"].append(path)
    print(f"已保存初始全局模型到 {path}")
    print(f"main_dict is {main_dict}")
    import threading
    t = [threading.Thread(target=client.run) for client in clients]
    _ = [i.start() for i in t]
    _ = [i.join for i in t]