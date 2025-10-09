# -*- coding:utf-8 -*-
# @FileName :lattice.py
# @Time :2025/10/9 14:33
# @Author :M2883b0

"""
- 基于格密码的动态群签名库 -
  格密码基础操作模块

  本模块实现了格密码相关的基础数学运算，包括：
  1. LWE (Learning With Errors) 问题相关操作
  2. 格基生成与操作
  3. 格密码中的随机数生成和采样
"""

import numpy as np
import random
import math
from typing import Tuple, List, Dict, Any, Optional


class LatticeParameters:
    """
    格密码参数类，存储格密码方案的参数
    """
    def __init__(self, n: int = 256, q: int = 8192, sigma: float = 3.2):
        """
        初始化格密码参数
        
        Args:
            n: 安全参数，格的维度
            q: 模数
            sigma: 错误分布的标准差
        """
        self.n = n  # 安全参数
        self.q = q  # 模数
        self.sigma = sigma  # 错误分布的标准差


class LatticeOperations:
    """
    格密码基础操作类，提供格密码相关的数学运算
    """
    def __init__(self, params: LatticeParameters):
        """
        初始化格密码操作类
        
        Args:
            params: 格密码参数
        """
        self.params = params
        
    def sample_gaussian(self, mu: float = 0.0) -> int:
        """
        从高斯分布中采样
        
        Args:
            mu: 均值
            
        Returns:
            采样得到的整数
        """
        # 使用中心极限定理近似高斯分布
        sum_val = 0
        for _ in range(12):
            sum_val += random.uniform(-1, 1)
        return int(round(mu + self.params.sigma * sum_val / 2))
    
    def generate_random_matrix(self, rows: int, cols: int) -> np.ndarray:
        """
        生成随机矩阵
        
        Args:
            rows: 矩阵行数
            cols: 矩阵列数
            
        Returns:
            随机矩阵
        """
        return np.random.randint(0, self.params.q, size=(rows, cols), dtype=np.int64)
    
    def generate_error_vector(self, length: int) -> np.ndarray:
        """
        生成错误向量
        
        Args:
            length: 向量长度
            
        Returns:
            错误向量
        """
        return np.array([self.sample_gaussian() for _ in range(length)], dtype=np.int64)
    
    def matrix_vector_mult(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        矩阵向量乘法（模q）
        
        Args:
            matrix: 矩阵
            vector: 向量
            
        Returns:
            结果向量（模q）
        """
        return np.dot(matrix, vector) % self.params.q
    
    def add_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        向量加法（模q）
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            结果向量（模q）
        """
        return (vec1 + vec2) % self.params.q
    
    def mod_q(self, x: int) -> int:
        """
        取模运算，确保结果在[-q/2, q/2)范围内
        
        Args:
            x: 整数
            
        Returns:
            取模后的结果
        """
        res = x % self.params.q
        if res > self.params.q // 2:
            res -= self.params.q
        return res
    
    def mod_q_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        向量取模运算
        
        Args:
            vec: 向量
            
        Returns:
            取模后的向量
        """
        return np.array([self.mod_q(x) for x in vec], dtype=np.int64)
    
    def norm(self, vec: np.ndarray) -> float:
        """
        计算向量的欧几里得范数
        
        Args:
            vec: 向量
            
        Returns:
            向量的范数
        """
        return np.linalg.norm(vec)
    
    def hash_to_vector(self, message: bytes, length: int) -> np.ndarray:
        """
        将消息哈希为向量
        
        Args:
            message: 输入消息
            length: 输出向量长度
            
        Returns:
            哈希后的向量
        """
        # 简化实现，实际应用中应使用安全的哈希函数
        random.seed(hash(message))
        return np.array([random.randint(0, self.params.q-1) for _ in range(length)], dtype=np.int64)