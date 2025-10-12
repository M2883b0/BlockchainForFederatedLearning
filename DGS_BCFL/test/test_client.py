# -*- coding:utf-8 -*-
# @FileName :test_client.py
# @Time :2025/10/9 20:25
# @Author :M2883b0

import torch

from DGS_BCFL.src import data_split
from DGS_BCFL.src.FederatedLearning.learner import CNNModel
from DGS_BCFL.src.client import Client
from DGS_BCFL.src.utils.logger import info
from DGS_BCFL.src.owner import Owner

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
    # 联邦学习轮数
    max_round = 2
    # 模型类
    model_class = CNNModel

    # 使用data_split模块中的函数获取PyTorch数据集
    train_dataset = data_split.get_mnist_pytorch_dataset(train=True)
    test_dataset = data_split.get_mnist_pytorch_dataset(train=False)
    # test
    # 使用正确的数据集对象创建数据加载器
    client_dataloaders = data_split.create_client_dataloaders(train_dataset, 5, 64) * 2
    # client_dataloaders = data_split.create_client_dataloaders(train_dataset, 10, 64)
    client_test_loader = data_split.create_client_dataloaders(test_dataset, 5, 64) * 2

    # 初始化管理者，并获得初始化字典
    owner = Owner(rotation_cycle=1, model_class=model_class)
    main_dict = owner.get_main_dict()

    # 初始化客户端以及身份
    clients = [owner]
    role_dict = main_dict["role"][0]
    for i in range(clients_num):
        client_name = f"client_{i+1}"
        client = Client(epochs=2, client_name=client_name, data_loader=client_dataloaders[i], test_loader=client_test_loader[i], ModelClass=model_class, main_dict=main_dict)
        owner.join(client_name)
        clients.append(client)


    import threading
    for _ in range(max_round):
        t = [threading.Thread(target=client.run) for client in clients]
        _ = [i.start() for i in t]
        _ = [i.join() for i in t]