# -*- coding:utf-8 -*-
# @FileName :Validator.py
# @Time :2025/10/9 10:08
# @Author :M2883b0

"""
- 区块链联邦学习系统 -
  梯度验证器

  本模块实现了联邦学习中的梯度验证功能，包括：
  1. 梯度有效性检查
  2. 异常梯度检测
  3. 梯度统计分析
  4. 梯度质量评估
  5. 模型性能验证（准确度变化验证）
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import copy
import time
from DGS_BCFL.src.utils import logger


class Validator:
    """
    联邦学习梯度验证器类，负责验证客户端提交的梯度的有效性和质量，
    包括验证梯度应用后模型性能是否提升或保持在可接受范围内。
    """
    
    def __init__(self, gradient_threshold: float = None, 
                 z_score_threshold: float = 3.0, 
                 check_nan_inf: bool = True, 
                 check_norm: bool = True, 
                 min_gradient_size: int = 0,
                 accuracy_threshold: float = 0.5,  # 准确度提升阈值（百分比）
                 performance_drop_threshold: float = 5.0):  # 性能下降容忍阈值（百分比）
        """
        初始化梯度验证器
        
        Args:
            gradient_threshold: 梯度范数阈值，用于异常检测，默认为None（不启用）
            z_score_threshold: Z-score阈值，用于基于统计的异常检测
            check_nan_inf: 是否检查NaN和无穷大值
            check_norm: 是否检查梯度范数
            min_gradient_size: 梯度参数的最小元素数量，用于基本验证
            accuracy_threshold: 准确度提升阈值（百分比），大于此值认为梯度有效
            performance_drop_threshold: 性能下降容忍阈值（百分比），小于此值可接受
        """
        self.gradient_threshold = gradient_threshold
        self.z_score_threshold = z_score_threshold
        self.check_nan_inf = check_nan_inf
        self.check_norm = check_norm
        self.min_gradient_size = min_gradient_size
        self.accuracy_threshold = accuracy_threshold
        self.performance_drop_threshold = performance_drop_threshold
        
        # 存储历史梯度统计信息，用于基于历史的异常检测
        self.history_statistics = {}
        
        # 存储全局模型和性能基准
        self.global_model = None
        self.base_performance = None
        self.test_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # logger.info(f'梯度验证器使用设备: {self.device}')
    
    def load_global_model(self, model: torch.nn.Module) -> None:
        """
        加载全局模型
        
        Args:
            model: 全局模型实例
        """
        self.global_model = copy.deepcopy(model).to(self.device)
        # logger.info(f'已加载全局模型，模型类型: {model.__class__.__name__}')

    def set_test_loader(self, test_loader: torch.utils.data.DataLoader) -> None:
        """
        设置测试数据加载器

        Args:
            test_loader: 测试数据加载器
        """
        self.test_loader = test_loader
        # logger.info('已设置测试数据加载器')

    def calculate_base_performance(self) -> Dict[str, float]:
        """
        计算基准性能指标，作为后续验证的参考

        Returns:
            Dict: 包含损失和准确率的字典
        """
        if self.global_model is None:
            raise ValueError('全局模型未加载，请先调用load_global_model方法')
        if self.test_loader is None:
            raise ValueError('测试数据加载器未设置，请先调用set_test_loader方法')

        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total

        self.base_performance = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total
        }

        # logger.info(f'已计算基准性能: 损失={avg_loss:.4f}, 准确率={accuracy:.2f}%')
        return self.base_performance

    def apply_gradients_and_validate_performance(self, gradients: Dict[str, torch.Tensor],
                                               learning_rate: float = 1) -> Dict[str, Any]:
        """
        应用梯度到全局模型并验证性能变化

        Args:
            gradients: 待验证的梯度字典
            learning_rate: 学习率，用于应用梯度

        Returns:
            Dict: 包含性能验证结果的字典
        """
        if self.global_model is None:
            raise ValueError('全局模型未加载，请先调用load_global_model方法')
        if self.test_loader is None:
            raise ValueError('测试数据加载器未设置，请先调用set_test_loader方法')
        if self.base_performance is None:
            self.calculate_base_performance()

        # 创建模型副本用于测试梯度效果
        temp_model = copy.deepcopy(self.global_model)

        # 应用梯度到临时模型
        with torch.no_grad():
            for name, param in temp_model.named_parameters():
                if name in gradients and param.requires_grad:
                    param.data += learning_rate * gradients[name].to(self.device)

        # 评估应用梯度后的模型性能
        temp_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = temp_model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        new_performance = {
            'loss': test_loss / len(self.test_loader),
            'accuracy': 100 * correct / total,
            'samples': total
        }

        # 计算性能变化
        accuracy_change = new_performance['accuracy'] - self.base_performance['accuracy']
        loss_change = new_performance['loss'] - self.base_performance['loss']

        # 判断性能是否可接受
        is_performance_acceptable = False
        if accuracy_change >= self.accuracy_threshold:
            # 准确度提升足够大，梯度有效
            is_performance_acceptable = True
            status = 'accept'
            reason = f'准确度提升 {accuracy_change:.2f}%，大于阈值 {self.accuracy_threshold}%'
        elif accuracy_change >= -self.performance_drop_threshold:
            # 准确度下降在可容忍范围内
            is_performance_acceptable = True
            status = 'tolerate'
            reason = f'准确度下降 {abs(accuracy_change):.2f}%，在容忍阈值 {self.performance_drop_threshold}% 范围内'
        else:
            # 准确度下降过多，梯度无效
            is_performance_acceptable = False
            status = 'reject'
            reason = f'准确度下降 {abs(accuracy_change):.2f}%，超过容忍阈值 {self.performance_drop_threshold}%'

        result = {
            'is_acceptable': is_performance_acceptable,
            'status': status,
            'reason': reason,
            'base_performance': self.base_performance,
            'new_performance': new_performance,
            'accuracy_change': accuracy_change,
            'loss_change': loss_change
        }

        # logger.info(f'性能验证结果: {status} - {reason}')
        return result

    def validate(self, gradients: Dict[str, torch.Tensor],
                validate_performance: bool = True,
                learning_rate: float = 0.01) -> Tuple[bool, Dict[str, Any]]:
        """
        验证梯度的有效性，包括梯度本身的有效性和应用后的性能变化

        Args:
            gradients: 待验证的梯度字典
            validate_performance: 是否验证应用梯度后的性能变化
            learning_rate: 学习率，用于性能验证时应用梯度

        Returns:
            Tuple[bool, Dict]: (验证是否通过, 验证结果详情)
        """
        result = {
            'is_valid': True,
            'reasons': [],
            'statistics': {},
            'invalid_params': [],
            'performance_validation': None
        }

        # 检查梯度是否为空
        if not gradients:
            result['is_valid'] = False
            result['reasons'].append('梯度为空')
            logger.warning('梯度验证失败：梯度为空')
            return False, result

        # 计算梯度统计信息
        stats = self._calculate_gradient_statistics(gradients)
        result['statistics'] = stats

        # 检查每个参数的梯度
        for name, grad in gradients.items():
            param_result = self._validate_param_gradient(name, grad)

            if not param_result['is_valid']:
                result['is_valid'] = False
                result['invalid_params'].append({
                    'name': name,
                    'reasons': param_result['reasons'],
                    'statistics': param_result['statistics']
                })

        # 如果有无效参数，添加到总体原因
        if result['invalid_params']:
            result['reasons'].append(f'发现{len(result["invalid_params"])}个无效参数梯度')
            logger.warning(f'梯度验证失败：发现{len(result["invalid_params"])}个无效参数梯度')

        # 基于历史数据进行异常检测（如果有历史数据）
        if result['is_valid'] and self.history_statistics:
            historical_result = self._detect_anomalies_using_history(gradients, stats)
            if not historical_result['is_valid']:
                result['is_valid'] = False
                result['reasons'].extend(historical_result['reasons'])

        # 验证性能变化（如果启用且梯度本身有效）
        if result['is_valid'] and validate_performance and self.global_model and self.test_loader:
            performance_result = self.apply_gradients_and_validate_performance(gradients, learning_rate)
            result['performance_validation'] = performance_result

            if not performance_result['is_acceptable']:
                result['is_valid'] = False
                result['reasons'].append(f'性能验证失败：{performance_result["reason"]}')
                logger.warning(f'梯度验证失败：性能验证不通过')

        # 更新历史统计信息（仅当验证通过时）
        if result['is_valid']:
            self._update_history_statistics(stats)
            # logger.info('梯度验证通过')

        return result['is_valid'], result

    def _validate_param_gradient(self, name: str, grad: torch.Tensor) -> Dict[str, Any]:
        """
        验证单个参数的梯度

        Args:
            name: 参数名称
            grad: 参数梯度

        Returns:
            Dict: 验证结果
        """
        result = {
            'is_valid': True,
            'reasons': [],
            'statistics': {}
        }

        # 计算参数梯度的统计信息
        grad_norm = torch.norm(grad).item()
        grad_size = grad.nelement()

        result['statistics'] = {
            'norm': grad_norm,
            'size': grad_size,
            'mean': torch.mean(grad).item(),
            'std': torch.std(grad).item(),
            'min': torch.min(grad).item(),
            'max': torch.max(grad).item()
        }

        # 检查梯度大小
        if grad_size < self.min_gradient_size:
            result['is_valid'] = False
            result['reasons'].append(f'梯度元素数量不足（{grad_size} < {self.min_gradient_size}）')
            logger.warning(f'参数 {name} 梯度验证失败：梯度元素数量不足')
            return result

        # 检查NaN和无穷大值
        if self.check_nan_inf:
            if torch.isnan(grad).any():
                result['is_valid'] = False
                result['reasons'].append('包含NaN值')
                logger.warning(f'参数 {name} 梯度验证失败：包含NaN值')

            if torch.isinf(grad).any():
                result['is_valid'] = False
                result['reasons'].append('包含无穷大值')
                logger.warning(f'参数 {name} 梯度验证失败：包含无穷大值')

            if not result['is_valid']:
                return result

        # 检查梯度范数
        if self.check_norm and self.gradient_threshold is not None:
            if grad_norm > self.gradient_threshold:
                result['is_valid'] = False
                result['reasons'].append(f'梯度范数超过阈值（{grad_norm:.4f} > {self.gradient_threshold}）')
                logger.warning(f'参数 {name} 梯度验证失败：梯度范数超过阈值')

        return result

    def _calculate_gradient_statistics(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        计算梯度的整体统计信息

        Args:
            gradients: 梯度字典

        Returns:
            Dict: 梯度统计信息
        """
        # 收集所有梯度值
        all_values = []
        all_norms = []

        for name, grad in gradients.items():
            all_values.extend(grad.cpu().numpy().flatten().tolist())
            all_norms.append(torch.norm(grad).item())

        # 计算统计信息
        all_values_np = np.array(all_values)

        return {
            'total_params': len(gradients),
            'total_elements': len(all_values),
            'mean_norm': np.mean(all_norms) if all_norms else 0,
            'std_norm': np.std(all_norms) if all_norms else 0,
            'min_norm': np.min(all_norms) if all_norms else 0,
            'max_norm': np.max(all_norms) if all_norms else 0,
            'mean_value': np.mean(all_values_np) if len(all_values_np) > 0 else 0,
            'std_value': np.std(all_values_np) if len(all_values_np) > 0 else 0,
            'min_value': np.min(all_values_np) if len(all_values_np) > 0 else 0,
            'max_value': np.max(all_values_np) if len(all_values_np) > 0 else 0,
            'percentile_95': np.percentile(np.abs(all_values_np), 95) if len(all_values_np) > 0 else 0
        }

    def _detect_anomalies_using_history(self, gradients: Dict[str, torch.Tensor],
                                       current_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用历史数据检测异常梯度

        Args:
            gradients: 当前梯度字典
            current_stats: 当前梯度统计信息

        Returns:
            Dict: 异常检测结果
        """
        result = {
            'is_valid': True,
            'reasons': []
        }

        # 基于整体梯度范数的Z-score检测
        if 'mean_norm' in self.history_statistics and 'std_norm' in self.history_statistics:
            mean_historical = self.history_statistics['mean_norm']
            std_historical = self.history_statistics['std_norm']

            if std_historical > 0:
                z_score = abs(current_stats['mean_norm'] - mean_historical) / std_historical
                if z_score > self.z_score_threshold:
                    result['is_valid'] = False
                    result['reasons'].append(f'梯度范数异常（Z-score = {z_score:.2f} > {self.z_score_threshold}）')
                    logger.warning(f'梯度验证失败：梯度范数异常，Z-score = {z_score:.2f}')

        # 检查参数数量是否有明显变化
        if 'total_params' in self.history_statistics:
            historical_params = self.history_statistics['total_params']
            current_params = current_stats['total_params']

            # 如果参数数量变化超过20%，认为异常
            if abs(current_params - historical_params) / max(1, historical_params) > 0.2:
                result['is_valid'] = False
                result['reasons'].append(f'参数数量变化异常（{current_params} vs 历史 {historical_params}）')
                logger.warning(f'梯度验证失败：参数数量变化异常')

        return result

    def _update_history_statistics(self, current_stats: Dict[str, Any]) -> None:
        """
        更新历史统计信息

        Args:
            current_stats: 当前梯度统计信息
        """
        # 简单的指数移动平均更新历史统计
        alpha = 0.1  # 平滑因子

        for key, value in current_stats.items():
            if key in self.history_statistics:
                # 使用指数移动平均更新
                self.history_statistics[key] = alpha * value + (1 - alpha) * self.history_statistics[key]
            else:
                # 第一次设置值
                self.history_statistics[key] = value

    def reset_history(self) -> None:
        """
        重置历史统计信息
        """
        self.history_statistics = {}
        # logger.info('梯度验证器历史统计信息已重置')
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """
        获取验证结果的摘要文本
        
        Args:
            validation_result: 验证结果字典
            
        Returns:
            str: 验证结果摘要
        """
        if validation_result['is_valid']:
            summary = "梯度验证通过\n"
            stats = validation_result['statistics']
            summary += f"  参数数量: {stats['total_params']}\n"
            summary += f"  梯度元素总数: {stats['total_elements']}\n"
            summary += f"  平均梯度范数: {stats['mean_norm']:.4f}\n"
            summary += f"  梯度值范围: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]\n"
            
            # 如果有性能验证结果，添加到摘要
            if validation_result['performance_validation']:
                perf = validation_result['performance_validation']
                summary += "\n  性能验证结果:\n"
                summary += f"    基准准确率: {perf['base_performance']['accuracy']:.2f}%\n"
                summary += f"    新准确率: {perf['new_performance']['accuracy']:.2f}%\n"
                summary += f"    准确率变化: {perf['accuracy_change']:+.2f}%\n"
                summary += f"    状态: {perf['status']} - {perf['reason']}\n"
        else:
            summary = "梯度验证失败\n"
            summary += f"  失败原因: {', '.join(validation_result['reasons'])}\n"
            if validation_result['invalid_params']:
                summary += f"  无效参数数量: {len(validation_result['invalid_params'])}\n"
                for param in validation_result['invalid_params'][:3]:  # 只显示前3个无效参数
                    summary += f"    - {param['name']}: {', '.join(param['reasons'])}\n"
                if len(validation_result['invalid_params']) > 3:
                    summary += f"    - ... 还有{len(validation_result['invalid_params']) - 3}个无效参数\n"
            
            # 如果有性能验证结果，添加到摘要
            if validation_result['performance_validation']:
                perf = validation_result['performance_validation']
                summary += "\n  性能验证结果:\n"
                summary += f"    基准准确率: {perf['base_performance']['accuracy']:.2f}%\n"
                summary += f"    新准确率: {perf['new_performance']['accuracy']:.2f}%\n"
                summary += f"    准确率变化: {perf['accuracy_change']:+.2f}%\n"
                summary += f"    状态: {perf['status']} - {perf['reason']}\n"
        
        return summary


if __name__ == '__main__':
    """
    验证器类的简单示例
    """
    import torchvision
    import torchvision.transforms as transforms
    from learner import CNNModel
    
    # 创建一个简单的测试用例
    def create_test_gradients(is_valid=True):
        """创建测试用的梯度字典"""
        model = CNNModel()
        gradients = {
            'conv1.weight': torch.randn_like(model.conv1.weight),
            'conv1.bias': torch.randn_like(model.conv1.bias),
            'conv2.weight': torch.randn_like(model.conv2.weight),
            'conv2.bias': torch.randn_like(model.conv2.bias),
            'fc1.weight': torch.randn_like(model.fc1.weight),
            'fc1.bias': torch.randn_like(model.fc1.bias),
            'fc2.weight': torch.randn_like(model.fc2.weight),
            'fc2.bias': torch.randn_like(model.fc2.bias)
        }
        
        if not is_valid:
            # 使梯度无效：添加NaN值
            gradients['conv2.weight'][0, 0, 0, 0] = float('nan')
        
        return gradients, model
    
    # 准备测试数据
    def prepare_test_data():
        """准备MNIST测试数据"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                 download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                                 shuffle=False, num_workers=0)
        return test_loader
    
    # 测试有效梯度
    logger.info("\n===== 测试有效梯度 =====")
    validator = Validator(gradient_threshold=20.0, accuracy_threshold=0.5)
    
    # 创建测试梯度和模型
    valid_gradients, test_model = create_test_gradients(is_valid=True)
    
    # 加载模型和测试数据
    validator.load_global_model(test_model)
    test_loader = prepare_test_data()
    validator.set_test_loader(test_loader)
    
    # 计算基准性能
    logger.info("计算基准性能...")
    validator.calculate_base_performance()
    
    # 验证梯度（包括性能验证）
    logger.info("验证梯度（包括性能验证）...")
    is_valid, result = validator.validate(valid_gradients, validate_performance=True, learning_rate=0.01)
    logger.info(validator.get_validation_summary(result))
    
    # 测试无效梯度
    logger.info("\n===== 测试无效梯度（包含NaN值）=====")
    invalid_gradients, _ = create_test_gradients(is_valid=False)
    is_valid, result = validator.validate(invalid_gradients, validate_performance=False)
    logger.info(validator.get_validation_summary(result))
    
    # 测试性能验证功能 - 模拟性能下降过多的情况
    logger.info("\n===== 测试性能验证（模拟性能下降过多）=====")
    # 创建一个新的验证器实例
    performance_validator = Validator(gradient_threshold=20.0, accuracy_threshold=0.5, performance_drop_threshold=1.0)
    performance_validator.load_global_model(test_model)
    performance_validator.set_test_loader(test_loader)
    performance_validator.calculate_base_performance()
    
    # 创建会导致性能大幅下降的梯度（过大的随机值）
    bad_gradients, _ = create_test_gradients(is_valid=True)
    for name in bad_gradients:
        bad_gradients[name] *= 10.0  # 放大梯度，可能导致性能下降
    
    logger.info("验证可能导致性能下降的梯度...")
    is_valid, result = performance_validator.validate(bad_gradients, validate_performance=True, learning_rate=0.01)
    logger.info(performance_validator.get_validation_summary(result))
