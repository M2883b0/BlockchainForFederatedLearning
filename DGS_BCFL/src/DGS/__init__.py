# -*- coding:utf-8 -*-
# @FileName :__init__.py
# @Time :2025/10/9 14:33
# @Author :M2883b0

"""
- 基于格密码的动态群签名库 -
  区块链联邦学习系统的密码学组件

  本库提供了基于格密码的动态群签名方案的完整实现，主要功能包括：
  1. 群初始化与密钥管理
  2. 成员加入与退出管理
  3. 群签名生成与验证
  4. 格密码基础操作

  使用示例：
  >>> from DGS import DynamicGroupSignature, LatticeParameters
  >>> # 创建签名方案实例
  >>> dgs = DynamicGroupSignature()
  >>> # 初始化群
  >>> dgs.setup()
  >>> # 成员加入群
  >>> member_key = dgs.join_group('member1')
  >>> # 生成签名
  >>> signature = dgs.sign(b'Hello World', 'member1')
  >>> # 验证签名
  >>> is_valid = dgs.verify(b'Hello World', signature)
  >>> info(f'签名有效: {is_valid}')
  True
"""

# 导入主要类和函数
import numpy as np
from typing import Dict, List, Any, Optional
from .lattice import LatticeParameters, LatticeOperations
from .dgs import DynamicGroupSignature
from ..utils.logger import setup_logger, info, debug, warning, error

# 定义版本信息
__version__ = '1.0.0'
__author__ = 'M2883b0'
__description__ = '基于格密码的动态群签名库，适用于区块链联邦学习系统'

# 导出公共API
__all__ = [
    'LatticeParameters',
    'LatticeOperations',
    'DynamicGroupSignature'
]

if __name__ == '__main__':
    # 简单演示
    info("===== 基于格密码的动态群签名库演示 =====")
    
    # 创建签名方案实例
    dgs = DynamicGroupSignature()
    
    # 初始化群
    info("1. 初始化群...")
    public_key = dgs.setup()
    info(f"   群已初始化，群ID: {public_key['group_id']}")
    
    # 成员加入群
    info("2. 成员加入群...")
    members = ['member1', 'member2', 'member3']
    for member in members:
        dgs.join_group(member)
        info(f"   成员 {member} 已加入群")
    
    # 生成签名
    info("3. 生成群签名...")
    message = "这是一个用于测试的消息".encode('utf-8')
    signature = dgs.sign(message, 'member1')
    info(f"   成员1 已为消息生成签名")
    
    # 验证签名
    info("4. 验证群签名...")
    is_valid = dgs.verify(message, signature)
    info(f"   签名验证结果: {'有效' if is_valid else '无效'}")
    
    # 撤销成员
    info("5. 撤销群成员...")
    dgs.revoke_member('member2')
    info(f"   成员2 已被撤销")
    
    # 尝试使用已撤销成员的ID签名
    info("6. 尝试使用已撤销成员的ID签名...")
    try:
        invalid_signature = dgs.sign(message, 'member2')
        error("   错误: 应该无法使用已撤销成员的ID签名")
    except ValueError as e:
        info(f"   预期行为: {e}")
    
    # 验证群信息
    info("7. 查看群信息...")
    group_info = dgs.get_group_info()
    info(f"   群信息: {group_info}")
    
    info("===== 演示结束 =====")
