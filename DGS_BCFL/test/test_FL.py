

# -*- coding:utf-8 -*-
# @FileName :test_FL.py
# @Time :2025/10/9
# @Author :M2883b0

"""
联邦学习系统测试脚本

此脚本包含完整的联邦学习测试流程，包括：
1. 数据加载与预处理
2. 全局模型初始化
3. 多客户端联邦学习训练
4. 模型性能评估与结果分析
5. 梯度有效性验证（包含性能验证）
"""

import torch
import torchvision
import torchvision.transforms as transforms
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import os

from DGS_BCFL.src.FederatedLearning.learner import FederatedLearner, MODEL_CHOICES
from DGS_BCFL.src.FederatedLearning.aggregator import Aggregator
from DGS_BCFL.src import data_split

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def run_federated_learning_test(
    model_type='cnn',  # 可选: 'simple', 'cnn', 'deep_cnn', 'mlp'
    num_clients=3,
    num_rounds=3,
    local_epochs=2,
    learning_rate=0.01,
    batch_size=32,
    gradient_threshold=10.0,
    accuracy_threshold=0.5,  # 准确度提升阈值（百分比）
    performance_drop_threshold=3.0,  # 性能下降容忍阈值（百分比）
    device=None
):
    """
    运行联邦学习测试
    
    Args:
        model_type: 模型类型
        num_clients: 客户端数量
        num_rounds: 联邦学习轮数
        local_epochs: 每个客户端本地训练轮数
        learning_rate: 学习率
        batch_size: 批次大小
        gradient_threshold: 梯度阈值
        accuracy_threshold: 准确度提升阈值（百分比）
        performance_drop_threshold: 性能下降容忍阈值（百分比）
        device: 计算设备
        
    Returns:
        Dict: 包含测试结果的字典
    """
    # 自动判断设备，如果未指定则优先使用GPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试脚本使用设备: {device}")
    
    # 选择模型类型
    print(f"选择的模型类型: {model_type}")
    model_class = MODEL_CHOICES.get(model_type, MODEL_CHOICES['simple'])
    global_model = model_class().to(device)
    
    # 数据预处理和加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    print("\n正在加载MNIST数据集...")
    start_time = time.time()
    
    # 加载MNIST训练集和测试集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 使用data_split模块中的函数为每个客户端分配数据集
    client_dataloaders = data_split.create_client_dataloaders(
        train_dataset=train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        shuffle=True,
        verbose=True
    )
    
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n=== 开始多客户端联邦学习测试 ===")
    print(f"模型名称: {model_class.__name__}")
    print(f"客户端数量: {num_clients}, 联邦学习轮数: {num_rounds}")
    print(f"每轮客户端本地训练轮数: {local_epochs}, 学习率: {learning_rate}")
    print(f"模型参数数量: {sum(p.numel() for p in global_model.parameters())}")
    
    # 初始化聚合器，包含梯度性能验证功能
    aggregator = Aggregator(
        global_model=global_model,
        learning_rate=learning_rate,
        gradient_threshold=gradient_threshold,
        device=device,
        test_loader=test_loader,  # 传入测试数据加载器用于性能验证
    )
    
    # 初始化联邦学习客户端
    clients = []
    for i in range(num_clients):
        # 每个客户端使用相同的全局模型初始化
        client_model = copy.deepcopy(global_model)
        client = FederatedLearner(
            client_model,
            client_dataloaders[i],
            learning_rate=learning_rate,
            epochs=local_epochs,
            device=device
        )
        clients.append(client)
    
    # 存储每轮的性能指标
    round_metrics = []
    # 存储每个客户端每轮的训练参数
    client_metrics = {}
    # 存储梯度验证结果
    validation_results = {}
    for r in range(1, num_rounds + 1):
        validation_results[r] = {}
    for i in range(num_clients):
        client_metrics[f'client_{i+1}'] = []
    
    # 评估初始全局模型性能
    initial_evaluator = FederatedLearner(copy.deepcopy(global_model), None, device=device)
    initial_results = initial_evaluator.evaluate(test_loader)
    print("\n初始全局模型性能:")
    print(f"损失: {initial_results['loss']:.4f}, 准确率: {initial_results['accuracy']:.2f}%")
    round_metrics.append({
        'round': 0,
        'loss': initial_results['loss'],
        'accuracy': initial_results['accuracy']
    })
    
    # 执行多轮联邦学习
    total_fl_start_time = time.time()
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== 联邦学习第 {round_num}/{num_rounds} 轮 ===")
        
        # 重置聚合器收集的梯度
        aggregator.reset_collected_gradients()
        
        # 每个客户端进行本地训练
        for client_idx, client in enumerate(clients):
            print(f"\n[客户端 {client_idx + 1}] 开始本地训练...")
            
            # 确保客户端使用最新的全局模型
            client.load_global_model(global_model)
            
            # 执行本地训练
            client_train_start_time = time.time()
            train_results = client.train()
            client_train_time = time.time() - client_train_start_time
            
            print(f"[客户端 {client_idx + 1}] 训练完成!")
            print(f"[客户端 {client_idx + 1}] 训练耗时: {client_train_time:.2f}秒")
            print(f"[客户端 {client_idx + 1}] 平均训练损失: {train_results['loss']:.4f}")
            print(f"[客户端 {client_idx + 1}] 训练准确率: {train_results['accuracy']:.2f}%")
            
            # 保存客户端本轮训练参数
            client_metric = {
                'round': round_num,
                'loss': train_results['loss'],
                'accuracy': train_results['accuracy'],
                'training_time': client_train_time,
                'samples': train_results['samples']
            }
            client_metrics[f'client_{client_idx + 1}'].append(client_metric)
            
            # 导出梯度并通过聚合器收集
            gradients = client.export_gradients()
            success = aggregator.collect_gradients(gradients)
            
            if success:
                print(f"[客户端 {client_idx + 1}] 梯度已成功收集")
            else:
                print(f"[客户端 {client_idx + 1}] 梯度收集失败，将不参与本轮聚合")
        
        # 使用聚合器聚合梯度
        aggregated_gradients = aggregator.aggregate(strategy='average')
        
        # 使用聚合器更新全局模型
        aggregator.update_global_model(aggregated_gradients)
        
        # 使用聚合器评估当前全局模型性能
        results = aggregator.evaluate_model(test_loader)
        print(f"\n[服务器] 第 {round_num} 轮全局模型性能:")
        print(f"[服务器] 损失: {results['loss']:.4f}, 准确率: {results['accuracy']:.2f}%")
        
        # 记录本轮性能指标
        round_metrics.append({
            'round': round_num,
            'loss': results['loss'],
            'accuracy': results['accuracy']
        })
    
    total_fl_time = time.time() - total_fl_start_time
    
    print(f"\n=== 多客户端联邦学习训练完成 ===")
    print(f"总训练耗时: {total_fl_time:.2f}秒")
    
    # 打印每轮性能指标
    print("\n各轮性能指标汇总:")
    for metrics in round_metrics:
        print(f"轮次 {metrics['round']}: 损失={metrics['loss']:.4f}, 准确率={metrics['accuracy']:.2f}%")
    
    # 比较初始和最终性能
    initial_accuracy = round_metrics[0]['accuracy']
    final_accuracy = round_metrics[-1]['accuracy']
    accuracy_improvement = final_accuracy - initial_accuracy
    print(f"\n准确率提升: {accuracy_improvement:.2f}%")
    
    print("\n提示: 您可以通过修改以下参数来调整联邦学习过程:\n"
          f"- num_clients: 当前设置为 {num_clients}\n"
          f"- num_rounds: 当前设置为 {num_rounds}\n"
          f"- local_epochs: 当前设置为 {local_epochs}\n"
          f"- model_type: 当前设置为 '{model_type}'\n"
          f"- accuracy_threshold: 当前设置为 {accuracy_threshold}%\n"
          f"- performance_drop_threshold: 当前设置为 {performance_drop_threshold}%")
    
    print("\n支持的模型类型:\n"
          "- 'simple': 简单的全连接神经网络\n"
          "- 'cnn': 简单的卷积神经网络\n"
          "- 'deep_cnn': 更深的卷积神经网络\n"
          "- 'mlp': 多层感知器")
    
    # 返回测试结果
    return {
        'model_type': model_type,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'total_training_time': total_fl_time,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'round_metrics': round_metrics,
        'client_metrics': client_metrics,
        'validation_results': validation_results,
        'accuracy_threshold': accuracy_threshold,
        'performance_drop_threshold': performance_drop_threshold
    }


def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'创建目录: {directory}')


def plot_client_metrics(results):
    """
    绘制每个客户端的训练指标图表并保存到文件
    
    Args:
        results: 包含测试结果的字典
    """
    # 确保实验结果目录存在
    ensure_directory_exists('./ExperimentalResults')
    
    client_metrics = results['client_metrics']
    num_clients = results['num_clients']
    num_rounds = results['num_rounds']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'联邦学习客户端性能指标 ({results["model_type"]}模型, {num_clients}个客户端, {num_rounds}轮)', fontsize=16)
    
    # 准备数据
    rounds = list(range(1, num_rounds + 1))
    
    # 绘制每个客户端的准确率曲线
    for client_id, metrics in client_metrics.items():
        accuracy = [m['accuracy'] for m in metrics]
        axes[0, 0].plot(rounds, accuracy, marker='o', label=client_id)
    axes[0, 0].set_title('客户端准确率对比')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('准确率 (%)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 绘制每个客户端的损失曲线
    for client_id, metrics in client_metrics.items():
        loss = [m['loss'] for m in metrics]
        axes[0, 1].plot(rounds, loss, marker='s', label=client_id)
    axes[0, 1].set_title('客户端损失对比')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 绘制每个客户端的训练时间
    for client_id, metrics in client_metrics.items():
        train_time = [m['training_time'] for m in metrics]
        axes[1, 0].plot(rounds, train_time, marker='^', label=client_id)
    axes[1, 0].set_title('客户端训练时间')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('时间 (秒)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 绘制全局模型准确率和客户端平均准确率对比
    global_accuracy = [m['accuracy'] for m in results['round_metrics'][1:]]  # 跳过初始轮次
    avg_client_accuracy = []
    for r in range(num_rounds):
        round_accuracies = []
        for client_id in client_metrics:
            if r < len(client_metrics[client_id]):
                round_accuracies.append(client_metrics[client_id][r]['accuracy'])
        avg_client_accuracy.append(np.mean(round_accuracies))
    
    axes[1, 1].plot(rounds, global_accuracy, marker='o', label='全局模型准确率')
    axes[1, 1].plot(rounds, avg_client_accuracy, marker='s', label='客户端平均准确率')
    axes[1, 1].set_title('全局模型与客户端平均准确率对比')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('准确率 (%)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 生成带时间戳的文件名
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'./ExperimentalResults/client_metrics_{results["model_type"]}_{num_clients}clients_{num_rounds}rounds_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f'客户端指标图表已保存到: {filename}')


def plot_round_metrics(results):
    """
    绘制每轮全局模型性能指标图表并保存到文件
    
    Args:
        results: 包含测试结果的字典
    """
    # 确保实验结果目录存在
    ensure_directory_exists('./ExperimentalResults')
    
    round_metrics = results['round_metrics']
    
    rounds = [m['round'] for m in round_metrics]
    loss = [m['loss'] for m in round_metrics]
    accuracy = [m['accuracy'] for m in round_metrics]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制损失曲线
    color = 'tab:red'
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失值', color=color)
    ax1.plot(rounds, loss, 'o-', color=color, label='损失')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建第二个y轴绘制准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('准确率 (%)', color=color)
    ax2.plot(rounds, accuracy, 's-', color=color, label='准确率')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加标题和网格
    plt.title(f'全局模型性能变化 (准确率提升: {results["accuracy_improvement"]:.2f}%)')
    ax1.grid(True)
    
    # 整合两个图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 生成带时间戳的文件名
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'./ExperimentalResults/global_metrics_{results["model_type"]}_{results["num_clients"]}clients_{results["num_rounds"]}rounds_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f'全局模型指标图表已保存到: {filename}')


def plot_validation_results(results, validation_results):
    """
    绘制梯度验证结果图表
    
    Args:
        results: 包含测试结果的字典
        validation_results: 包含验证结果的字典，格式为 {轮次: {客户端ID: 验证结果}}
    """
    # 确保实验结果目录存在
    ensure_directory_exists('./ExperimentalResults')
    
    if not validation_results:
        print("警告：没有验证结果数据可供绘制")
        return
    
    num_rounds = results['num_rounds']
    rounds = list(range(1, num_rounds + 1))
    
    # 准备数据
    valid_gradients_count = {r: 0 for r in rounds}
    invalid_gradients_count = {r: 0 for r in rounds}
    accuracy_changes = []
    round_labels = []
    
    # 统计每轮的有效/无效梯度数量
    for round_num, clients in validation_results.items():
        if round_num in rounds:
            for client_id, result in clients.items():
                if result.get('is_valid', True):
                    valid_gradients_count[round_num] += 1
                else:
                    invalid_gradients_count[round_num] += 1
                
                # 收集性能验证结果
                if 'performance_validation' in result and 'accuracy_change' in result['performance_validation']:
                    accuracy_changes.append(result['performance_validation']['accuracy_change'])
                    round_labels.append(f'轮{round_num}_{client_id}')
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'联邦学习梯度验证结果分析', fontsize=16)
    
    # 绘制有效/无效梯度分布
    x = np.arange(len(rounds))
    width = 0.35
    
    valid_counts = [valid_gradients_count[r] for r in rounds]
    invalid_counts = [invalid_gradients_count[r] for r in rounds]
    
    axes[0].bar(x - width/2, valid_counts, width, label='有效梯度')
    axes[0].bar(x + width/2, invalid_counts, width, label='无效梯度')
    
    axes[0].set_xlabel('轮次')
    axes[0].set_ylabel('梯度数量')
    axes[0].set_title('各轮有效/无效梯度分布')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rounds)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 添加每个柱子的数值标签
    for i, v in enumerate(valid_counts):
        axes[0].text(i - width/2, v + 0.1, str(v), ha='center')
    for i, v in enumerate(invalid_counts):
        axes[0].text(i + width/2, v + 0.1, str(v), ha='center')
    
    # 绘制准确度变化柱状图
    if accuracy_changes and round_labels:
        # 设置阈值参考线
        accuracy_threshold = results.get('accuracy_threshold', 0.5) if isinstance(results, dict) else 0.5
        performance_drop_threshold = results.get('performance_drop_threshold', 3.0) if isinstance(results, dict) else 3.0
        
        # 创建颜色映射
        colors = []
        for change in accuracy_changes:
            if change >= accuracy_threshold:
                colors.append('green')  # 准确度提升足够大
            elif change >= -performance_drop_threshold:
                colors.append('blue')   # 准确度下降在可容忍范围内
            else:
                colors.append('red')    # 准确度下降过多
        
        axes[1].bar(round_labels, accuracy_changes, color=colors)
        axes[1].axhline(y=accuracy_threshold, color='g', linestyle='--', label=f'准确度提升阈值 ({accuracy_threshold}%)')
        axes[1].axhline(y=-performance_drop_threshold, color='r', linestyle='--', label=f'性能下降阈值 (-{performance_drop_threshold}%)')
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        axes[1].set_xlabel('轮次_客户端')
        axes[1].set_ylabel('准确度变化 (%)')
        axes[1].set_title('梯度应用后的准确度变化')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 设置Y轴范围，确保能看到所有数据
        min_change = min(accuracy_changes + [-performance_drop_threshold * 1.5])
        max_change = max(accuracy_changes + [accuracy_threshold * 1.5])
        axes[1].set_ylim(min_change - 0.5, max_change + 0.5)
        
        # 添加每个柱子的数值标签
        for i, v in enumerate(accuracy_changes):
            axes[1].text(i, v + 0.1 if v >= 0 else v - 0.3, f'{v:.2f}%', ha='center', rotation=90)
    else:
        axes[1].text(0.5, 0.5, '没有可用的性能验证数据', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xlabel('轮次_客户端')
        axes[1].set_ylabel('准确度变化 (%)')
        axes[1].set_title('梯度应用后的准确度变化')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 生成带时间戳的文件名
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'./ExperimentalResults/validation_results_{results["model_type"]}_{results["num_clients"]}clients_{results["num_rounds"]}rounds_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f'梯度验证结果图表已保存到: {filename}')


if __name__ == '__main__':
    """
    联邦学习系统测试主函数
    
    此函数提供了一个简单的入口点，可以通过修改参数来测试不同配置下的联邦学习性能。
    """
    # 运行标准测试配置，包含梯度性能验证
    test_results = run_federated_learning_test(
        model_type='cnn',        # 模型类型
        num_clients=2,           # 客户端数量
        num_rounds=2,            # 联邦学习轮数
        local_epochs=2,          # 本地训练轮数
        learning_rate=0.1,       # 学习率
        batch_size=32,           # 批次大小
        gradient_threshold=10.0, # 梯度阈值
        accuracy_threshold=0.5,  # 准确度提升阈值（百分比）
        performance_drop_threshold=3.0  # 性能下降容忍阈值（百分比）
    )
    
    # 绘制每个客户端的训练参数图表
    print("正在绘制客户端训练参数图表...")
    plot_client_metrics(test_results)
    
    # 绘制全局模型每轮性能指标图表
    print("正在绘制全局模型性能指标图表...")
    plot_round_metrics(test_results)
    
    # 绘制梯度验证结果图表
    print("正在绘制梯度验证结果图表...")
    if 'validation_results' in test_results:
        plot_validation_results(test_results, test_results['validation_results'])
    
    # 如果需要运行不同配置的测试，可以取消下面的注释
    
    # 测试不同模型类型
    # test_cnn = run_federated_learning_test(model_type='cnn')
    # test_simple = run_federated_learning_test(model_type='simple')
    # test_deep_cnn = run_federated_learning_test(model_type='deep_cnn')
    # test_mlp = run_federated_learning_test(model_type='mlp')
    
    # 测试不同客户端数量
    # test_2clients = run_federated_learning_test(num_clients=2)
    # test_5clients = run_federated_learning_test(num_clients=5)
    
    # 测试不同学习率
    # test_lr005 = run_federated_learning_test(learning_rate=0.05)
    # test_lr0001 = run_federated_learning_test(learning_rate=0.001)
    
    # 测试不同的性能验证阈值
    # test_high_accuracy_threshold = run_federated_learning_test(accuracy_threshold=1.0)
    # test_low_performance_threshold = run_federated_learning_test(performance_drop_threshold=5.0)
