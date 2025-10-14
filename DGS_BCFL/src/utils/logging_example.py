# -*- coding:utf-8 -*-
# @FileName :logging_example.py
# @Time :2024/10/10
# @Author :M2883b0

"""
- 区块链联邦学习系统 -
  日志配置示例

  本模块提供了如何使用统一的logging配置的示例，包括：
  1. 日志输出到终端的配置
  2. 日志输出到文件的配置
  3. 日志同时输出到终端和文件的配置
  4. 不同日志级别的使用示例
"""

import os
from .logger import setup_logger


def setup_terminal_logging(level='INFO'):
    """
    配置日志输出到终端
    
    参数:
        level: 日志级别，可选值: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    setup_logger(
        log_file=None,  # 不输出到文件
        log_level=level,
        console=True,   # 输出到终端
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_file_logging(log_file='federated_learning.log', level='INFO'):
    """
    配置日志输出到文件
    
    参数:
        log_file: 日志文件路径
        level: 日志级别，可选值: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    # 确保日志文件目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    setup_logger(
        log_file=log_file,
        log_level=level,
        console=False,  # 不输出到终端
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_both_logging(log_file='federated_learning.log', level='INFO', console_level='INFO'):
    """
    配置日志同时输出到终端和文件，可设置不同的日志级别
    
    参数:
        log_file: 日志文件路径
        level: 文件日志级别
        console_level: 终端日志级别
    """
    # 确保日志文件目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    setup_logger(
        log_file=log_file,
        log_level=level,
        console=True,
        console_level=console_level,
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        console_format='%(levelname)s: %(message)s'  # 终端输出简化格式
    )


def example_usage():
    """
    日志配置使用示例
    """
    # 从logger模块导入日志函数
    from .logger import info, debug, warning, error, critical
    
    print("===== 日志配置示例 =====")
    
    # 示例1: 配置日志输出到终端
    print("\n1. 配置日志输出到终端:")
    setup_terminal_logging(level='DEBUG')
    
    debug("这是一条DEBUG级别日志")
    info("这是一条INFO级别日志")
    warning("这是一条WARNING级别日志")
    error("这是一条ERROR级别日志")
    critical("这是一条CRITICAL级别日志")
    
    # 示例2: 配置日志输出到文件
    print("\n2. 配置日志输出到文件 'logs/fl_system.log':")
    setup_file_logging(log_file='logs/fl_system.log', level='INFO')
    print("   日志已配置为输出到文件 'logs/fl_system.log'")
    
    # 示例3: 配置日志同时输出到终端和文件
    print("\n3. 配置日志同时输出到终端和文件:")
    setup_both_logging(
        log_file='logs/fl_system_combined.log',
        level='DEBUG',          # 文件记录所有DEBUG及以上级别
        console_level='INFO'    # 终端只显示INFO及以上级别
    )
    print("   日志已配置为同时输出到终端和文件 'logs/fl_system_combined.log'")
    print("   - 文件记录级别: DEBUG")
    print("   - 终端显示级别: INFO")
    
    debug("这条DEBUG日志只会出现在文件中")
    info("这条INFO日志会同时出现在终端和文件中")


if __name__ == '__main__':
    example_usage()