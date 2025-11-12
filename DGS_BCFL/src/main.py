# -*- coding:utf-8 -*-
# @FileName :test_client.py
# @Time :2025/10/9 20:25
# @Author :M2883b0
import time

if __name__ == '__main__':
    from FederatedLearning.learner import CNNModel

    clients_num = 20  # 客户端总数
    rotation_cycle = 2  # 角色轮换间隔轮数
    max_round = 100   # 联邦学习轮数
    learning_rate = 0.01
    model_class = CNNModel  # 选用CNN模型

    no_bad_train = False
    bad_10_train = True

    start = time.time()
    if no_bad_train:
        import data_split

        # 使用data_split模块中的函数获取PyTorch数据集
        train_dataset = data_split.get_mnist_pytorch_dataset(train=True)
        test_dataset = data_split.get_mnist_pytorch_dataset(train=False)
        # test
        # 使用正确的数据集对象创建数据加载器
        client_dataloaders = data_split.create_client_dataloaders(train_dataset, clients_num, 64, num_workers=2)
        client_test_loader = []
        test_data_loader = data_split.create_client_dataloaders(train_dataset, 1, 64, num_workers=2)[0]
        tmp = 3
        for _ in range(tmp+1):
            t = data_split.create_client_dataloaders(test_dataset, clients_num // tmp, 64, num_workers=2)
            client_test_loader.extend(t)
        # 无恶意客户端
        from owner import Owner
        from client import Client, BadClient

        # 初始化管理者，并获得初始化字典
        owner = Owner(rotation_cycle=rotation_cycle, model_class=model_class)
        main_dict = owner.get_main_dict()

        # 初始化客户端以及身份
        clients = [owner]
        for i in range(clients_num):
            client_name = f"client_{i + 1}"
            client = Client(epochs=2, client_name=client_name, data_loader=client_dataloaders[i],
                            test_loader=client_test_loader[i], ModelClass=model_class, main_dict=main_dict, lock=owner.get_lock(),
                            learning_rate=learning_rate, merge_data_loader=test_data_loader)
            owner.join(client_name)
            clients.append(client)

        import threading

        for _ in range(max_round):
            t = [threading.Thread(target=client.run) for client in clients]
            _ = [i.start() for i in t]
            _ = [i.join() for i in t]

        # 保存实验过程和结果
        import pickle

        with open(f"main_dict_{clients_num}_{0}.pkl", "wb") as f:
            pickle.dump(main_dict, f)
        print(f"无恶意客户端耗时{time.time() - start}")
    if bad_10_train:
        # 有恶意客户端
        import data_split

        # 使用data_split模块中的函数获取PyTorch数据集
        train_dataset = data_split.get_mnist_pytorch_dataset(train=True)
        test_dataset = data_split.get_mnist_pytorch_dataset(train=False)
        # test
        # 使用正确的数据集对象创建数据加载器
        client_dataloaders = data_split.create_client_dataloaders(train_dataset, clients_num, 64, num_workers=2)
        client_test_loader = []
        test_data_loader = data_split.create_client_dataloaders(test_dataset, 1, 64, num_workers=2)[0]
        tmp = 1
        for _ in range(tmp):
            t = data_split.create_client_dataloaders(test_dataset, clients_num // tmp, 64, num_workers=2)
            client_test_loader.extend(t)
        from owner import Owner
        from client import Client, BadClient

        bad_percent = 0.1
        bad_num = int(clients_num * bad_percent)

        # 初始化管理者，并获得初始化字典
        owner = Owner(rotation_cycle=rotation_cycle, model_class=model_class)
        main_dict = owner.get_main_dict()
        main_lock = owner.get_lock()

        # 初始化客户端以及身份
        clients = [owner]
        for i in range(clients_num - bad_num):
            client_name = f"client_{i + 1}"
            client = Client(epochs=2, client_name=client_name, data_loader=client_dataloaders[i], lock=main_lock,
                            test_loader=client_test_loader[i], ModelClass=model_class, main_dict=main_dict,
                            learning_rate=learning_rate, base_path=".", merge_data_loader=test_data_loader)
            owner.join(client_name)
            clients.append(client)
        for i in range(bad_num):
            client_name = f"bad_client_{i + 1}"
            client = BadClient(epochs=2, client_name=client_name, data_loader=client_dataloaders[i],
                               test_loader=client_test_loader[i], ModelClass=model_class, main_dict=main_dict, lock=main_lock,
                               learning_rate=learning_rate, base_path=".",  merge_data_loader=test_data_loader)
            owner.join(client_name)
            clients.append(client)

        import threading

        for _ in range(max_round):
            t = [threading.Thread(target=client.run) for client in clients]
            _ = [i.start() for i in t]
            _ = [i.join() for i in t]

        # 保存实验过程和结果
        import pickle

        with open(f"main_dict_{clients_num}_{bad_percent * 100}.pkl", "wb") as f:
            pickle.dump(main_dict, f)

    print(f"有恶意客户端耗时{time.time() - start}")