"""
 - Blockchain for Federated Learning -
      Federated Learning Script (TensorFlow 2.x Version)
"""

import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import pickle

def reset():
    # TF2 不再需要显式重置计算图
    pass

class NNWorker:
    def __init__(self, X=None, Y=None, tX=None, tY=None, size=0, Id="nn0", steps=10):
        self.id = Id
        self.train_x = X
        self.train_y = Y
        self.test_x = tX
        self.test_y = tY
        self.size = size
        self.learning_rate = 0.1
        self.num_steps = steps
        self.n_hidden_1 = 256
        self.n_hidden_2 = 256
        self.num_input = 784
        self.num_classes = 10
        # TF2 不再需要会话
        self.model = None

    def build(self, base):
        """构建基于基础权重的模型"""
        # 创建模型
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(self.n_hidden_1, activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  name='dense1'),
            tf.keras.layers.Dense(self.n_hidden_2, activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  name='dense2'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  name='output')
        ])
        self.model.get_layer('dense1').set_weights([base['w1'], base['b1']])
        self.model.get_layer('dense2').set_weights([base['w2'], base['b2']])
        self.model.get_layer('output').set_weights([base['wo'], base['bo']])

        # 编译模型
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy',
                           # loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def build_base(self):
        """构建随机初始化的基础模型"""

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(self.n_hidden_1, activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  name='dense1'),
            tf.keras.layers.Dense(self.n_hidden_2, activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  name='dense2'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros',
                                  name='output')
        ])

        # 编译模型
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy',
                           # loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        """训练模型"""
        # 使用整个数据集进行训练
        history = self.model.fit(
            self.train_x, self.train_y,
            epochs=self.num_steps,
            batch_size=len(self.train_x),  # 使用整个数据集作为批次
            verbose=0
        )

        # 打印训练进度
        for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
            print(f"Step {epoch+1}, Minibatch Loss= {loss:.4f}, Training Accuracy= {acc:.3f}")

        print("Optimization Finished!")

    def centralized_accuracy(self):
        """计算中心化训练精度"""
        cntz_acc = {'epoch': [], 'accuracy': []}

        # 构建基础模型
        self.build_base()

        # 逐步训练并记录精度
        for step in range(1, self.num_steps + 1):
            self.model.fit(
                self.train_x, self.train_y,
                epochs=1,
                batch_size=len(self.train_x),
                verbose=0
            )

            acc = self.evaluate()
            cntz_acc['epoch'].append(step)
            cntz_acc['accuracy'].append(acc)
            print(f"epoch {step}, accuracy {acc:.4f}")

        return cntz_acc

    def evaluate(self):
        """评估模型在测试集上的精度"""
        if self.model is None:
            raise ValueError("Model not built. Call build() or build_base() first.")
        # assert isinstance(self.model, keras.Model)
        _, accuracy = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        return accuracy

    def get_model(self):
        """获取模型权重"""
        if self.model is None:
            raise ValueError("Model not built. Call build() or build_base() first.")

        weights = self.model.get_weights()
        return {
            'w1': weights[0],
            'b1': weights[1],
            'w2': weights[2],
            'b2': weights[3],
            'wo': weights[4],
            'bo': weights[5],
            'size': self.size
        }

    def close(self):
        """关闭模型资源"""
        # TF2 不再需要显式关闭会话
        # 可以删除模型以释放资源
        del self.model
        self.model = None