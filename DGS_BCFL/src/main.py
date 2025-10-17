# -*- coding:utf-8 -*-
# @FileName :test_client.py
# @Time :2025/10/9 20:25
# @Author :M2883b0

if __name__ == '__main__':
    from FederatedLearning.learner import CNNModel
    clients_num = 10
    # 联邦学习轮数
    max_round = 2
    # 模型类
    model_class = CNNModel

    import data_split

    # 使用data_split模块中的函数获取PyTorch数据集
    train_dataset = data_split.get_mnist_pytorch_dataset(train=True)
    test_dataset = data_split.get_mnist_pytorch_dataset(train=False)
    # test
    # 使用正确的数据集对象创建数据加载器
    client_dataloaders = data_split.create_client_dataloaders(train_dataset, clients_num // 2, 64) * 2
    client_test_loader = data_split.create_client_dataloaders(test_dataset, clients_num // 2, 64) * 2



    from owner import Owner
    from client import Client

    # 初始化管理者，并获得初始化字典
    owner = Owner(rotation_cycle=1, model_class=model_class)
    main_dict = owner.get_main_dict()

    # 初始化客户端以及身份
    clients = [owner]
    for i in range(clients_num):
        client_name = f"client_{i + 1}"
        client = Client(epochs=2, client_name=client_name, data_loader=client_dataloaders[i],
                        test_loader=client_test_loader[i], ModelClass=model_class, main_dict=main_dict)
        owner.join(client_name)
        clients.append(client)

    import threading

    for _ in range(max_round):
        t = [threading.Thread(target=client.run) for client in clients]
        _ = [i.start() for i in t]
        _ = [i.join() for i in t]


    # 展示结果
