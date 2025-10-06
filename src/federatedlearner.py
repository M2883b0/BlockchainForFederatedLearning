import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader

def reset():
    # PyTorch 不需要显式重置计算图
    pass

class NNWorker:
    def __init__(self, X=None, Y=None, tX=None, tY=None, size=0, Id="nn0", steps=10):
        self.id = Id
        self.train_x = torch.tensor(X, dtype=torch.float32) if X is not None else None
        self.train_y = torch.tensor(Y, dtype=torch.long) if Y is not None else None
        self.test_x = torch.tensor(tX, dtype=torch.float32) if tX is not None else None
        self.test_y = torch.tensor(tY, dtype=torch.long) if tY is not None else None
        self.size = size
        self.learning_rate = 0.01
        self.num_steps = steps
        self.n_hidden_1 = 256
        self.n_hidden_2 = 256
        self.num_input = 784
        self.num_classes = 10
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, base):
        """构建基于基础权重的模型"""
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_input, self.n_hidden_1),
            nn.ReLU(),
            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU(),
            nn.Linear(self.n_hidden_2, self.num_classes)
        ).to(self.device)

        # 设置权重
        state_dict = {
            '1.weight': torch.tensor(base['w1']).t().to(self.device),
            '1.bias': torch.tensor(base['b1']).to(self.device),
            '3.weight': torch.tensor(base['w2']).t().to(self.device),
            '3.bias': torch.tensor(base['b2']).to(self.device),
            '5.weight': torch.tensor(base['wo']).t().to(self.device),
            '5.bias': torch.tensor(base['bo']).to(self.device)
        }
        self.model.load_state_dict(state_dict)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def build_base(self):
        """构建随机初始化的基础模型"""
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_input, self.n_hidden_1),
            nn.ReLU(),
            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU(),
            nn.Linear(self.n_hidden_2, self.num_classes)
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """训练模型"""
        if self.train_x is None or self.train_y is None:
            raise ValueError("Training data not provided")

        # 创建数据集和数据加载器
        dataset = TensorDataset(self.train_x, self.train_y)
        # 使用整个数据集作为批次
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        self.model.train()
        for epoch in range(self.num_steps):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 打印训练进度
            print(f"Step {epoch+1}, Minibatch Loss= {loss.item():.4f}")

        print("Optimization Finished!")

    def centralized_accuracy(self):
        """计算中心化训练精度"""
        cntz_acc = {'epoch': [], 'accuracy': []}

        # 构建基础模型
        self.build_base()

        # 创建数据集和数据加载器
        dataset = TensorDataset(self.train_x, self.train_y)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        self.model.train()
        for step in range(1, self.num_steps + 1):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 评估精度
            acc = self.evaluate()
            cntz_acc['epoch'].append(step)
            cntz_acc['accuracy'].append(acc)
            print(f"epoch {step}, accuracy {acc:.4f}")

        return cntz_acc

    def evaluate(self):
        """评估模型在测试集上的精度"""
        if self.model is None:
            raise ValueError("Model not built. Call build() or build_base() first.")
        if self.test_x is None or self.test_y is None:
            raise ValueError("Test data not provided")

        self.model.eval()
        with torch.no_grad():
            inputs, labels = self.test_x.to(self.device), self.test_y.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(labels)

        return accuracy

    def get_model(self):
        """获取模型权重"""
        if self.model is None:
            raise ValueError("Model not built. Call build() or build_base() first.")

        # 注意：PyTorch 权重矩阵是转置的，所以需要转置回来以匹配 TensorFlow 格式
        return {
            'w1': self.model[1].weight.data.t().cpu().numpy(),
            'b1': self.model[1].bias.data.cpu().numpy(),
            'w2': self.model[3].weight.data.t().cpu().numpy(),
            'b2': self.model[3].bias.data.cpu().numpy(),
            'wo': self.model[5].weight.data.t().cpu().numpy(),
            'bo': self.model[5].bias.data.cpu().numpy(),
            'size': self.size
        }

    def close(self):
        """关闭模型资源"""
        del self.model
        self.model = None