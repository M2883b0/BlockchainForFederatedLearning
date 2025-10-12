# -*- coding:utf-8 -*-
# @FileName :test_dgs.py
# @Time :2025/10/9 14:33
# @Author :M2883b0

"""
- 基于格密码的动态群签名库 -
  测试模块

  本模块包含了对动态群签名库的功能测试，包括：
  1. 群初始化测试
  2. 成员管理测试
  3. 签名生成与验证测试
  4. 安全性测试
"""

import unittest
import numpy as np
from . import DynamicGroupSignature, LatticeParameters
from ..utils.logger import setup_logger, info, debug, warning, error


class TestDynamicGroupSignature(unittest.TestCase):
    """
    动态群签名测试类
    """
    def setUp(self):
        """
        每个测试用例前的设置
        """
        # 创建默认参数的签名方案
        self.dgs_default = DynamicGroupSignature()
        
        # 创建自定义参数的签名方案
        self.params_custom = LatticeParameters(n=128, q=4096, sigma=3.0)
        self.dgs_custom = DynamicGroupSignature(self.params_custom)
        
        # 初始化群
        self.dgs_default.setup()
        self.dgs_custom.setup()
        
        # 测试用的成员ID
        self.member_ids = ['member1', 'member2', 'member3']
        
        # 测试用的消息
        self.test_messages = [
            "这是第一条测试消息".encode('utf-8'),
            "这是第二条测试消息".encode('utf-8'),
            b''  # 空消息
        ]
    
    def test_setup(self):
        """
        测试群初始化功能
        """
        # 验证默认参数的群初始化
        self.assertIsNotNone(self.dgs_default.group_public_key)
        self.assertIsNotNone(self.dgs_default.group_manager_key)
        self.assertEqual(self.dgs_default.group_public_key['group_id'], 'DGS_Group')
        
        # 验证自定义参数的群初始化
        self.assertIsNotNone(self.dgs_custom.group_public_key)
        self.assertIsNotNone(self.dgs_custom.group_manager_key)
        self.assertEqual(self.dgs_custom.group_public_key['params']['n'], 128)
        self.assertEqual(self.dgs_custom.group_public_key['params']['q'], 4096)
    
    def test_join_group(self):
        """
        测试成员加入群功能
        """
        # 测试默认参数的成员加入
        for member_id in self.member_ids:
            member_key = self.dgs_default.join_group(member_id)
            self.assertIsNotNone(member_key)
            self.assertEqual(member_key['member_id'], member_id)
            self.assertTrue(member_id in self.dgs_default.group_members)
        
        # 测试重复加入
        with self.assertRaises(ValueError):
            self.dgs_default.join_group(self.member_ids[0])
        
        # 测试自定义参数的成员加入
        for member_id in self.member_ids:
            member_key = self.dgs_custom.join_group(member_id)
            self.assertIsNotNone(member_key)
            self.assertEqual(member_key['member_id'], member_id)
    
    def test_sign_verify(self):
        """
        测试签名生成与验证功能
        """
        # 先加入成员
        for member_id in self.member_ids:
            self.dgs_default.join_group(member_id)
        
        # 测试签名生成与验证
        for message in self.test_messages:
            for member_id in self.member_ids:
                # 生成签名
                signature = self.dgs_default.sign(message, member_id)
                self.assertIsNotNone(signature)
                
                # 验证签名
                is_valid = self.dgs_default.verify(message, signature)
                self.assertTrue(is_valid, f"签名验证失败: 消息={message}, 成员={member_id}")
                
                # 测试篡改消息后的验证
                if message:
                    # 对消息进行微小修改
                    modified_message = message[:-1] if len(message) > 1 else "篡改的消息".encode('utf-8')
                    is_valid = self.dgs_default.verify(modified_message, signature)
                    self.assertFalse(is_valid, "篡改的消息应该验证失败")
    
    def test_revoke_member(self):
        """
        测试成员撤销功能
        """
        # 先加入成员
        for member_id in self.member_ids:
            self.dgs_default.join_group(member_id)
        
        # 撤销成员
        member_to_revoke = self.member_ids[1]
        result = self.dgs_default.revoke_member(member_to_revoke)
        self.assertTrue(result)
        self.assertFalse(member_to_revoke in self.dgs_default.group_members)
        
        # 尝试使用已撤销成员的ID签名
        with self.assertRaises(ValueError):
            self.dgs_default.sign(self.test_messages[0], member_to_revoke)
        
        # 尝试撤销不存在的成员
        result = self.dgs_default.revoke_member('non_existent_member')
        self.assertFalse(result)
    
    def test_group_info(self):
        """
        测试群信息查询功能
        """
        # 先加入成员
        for member_id in self.member_ids:
            self.dgs_default.join_group(member_id)
        
        # 查询群信息
        group_info = self.dgs_default.get_group_info()
        self.assertEqual(group_info['group_id'], 'DGS_Group')
        self.assertEqual(group_info['member_count'], len(self.member_ids))
        self.assertEqual(set(group_info['members']), set(self.member_ids))
        
        # 撤销一个成员后再查询
        self.dgs_default.revoke_member(self.member_ids[0])
        group_info = self.dgs_default.get_group_info()
        self.assertEqual(group_info['member_count'], len(self.member_ids) - 1)
        self.assertFalse(self.member_ids[0] in group_info['members'])
    
    def test_key_export_import(self):
        """
        测试密钥导入导出功能
        """
        # 导出公钥
        pk_bytes = self.dgs_default.export_public_key()
        self.assertIsNotNone(pk_bytes)
        self.assertTrue(isinstance(pk_bytes, bytes))
        
        # 创建新的签名方案实例
        new_dgs = DynamicGroupSignature()
        
        # 导入公钥
        new_dgs.import_public_key(pk_bytes)
        
        # 验证导入的公钥
        self.assertIsNotNone(new_dgs.group_public_key)
        self.assertEqual(new_dgs.group_public_key['group_id'], self.dgs_default.group_public_key['group_id'])
        self.assertEqual(new_dgs.group_public_key['params'], self.dgs_default.group_public_key['params'])
        # 验证矩阵内容是否相同
        self.assertTrue(np.array_equal(new_dgs.group_public_key['A'], self.dgs_default.group_public_key['A']))
        self.assertTrue(np.array_equal(new_dgs.group_public_key['B'], self.dgs_default.group_public_key['B']))
    
    def test_invalid_signature(self):
        """
        测试无效签名的验证
        """
        # 先加入成员
        for member_id in self.member_ids:
            self.dgs_default.join_group(member_id)
        
        # 生成有效签名
        signature = self.dgs_default.sign(self.test_messages[0], self.member_ids[0])
        
        # 修改签名中的字段
        modified_signature = signature.copy()
        
        # 修改a字段
        if 'a' in modified_signature:
            modified_signature['a'] = modified_signature['a'] + 1
        
        # 验证修改后的签名
        is_valid = self.dgs_default.verify(self.test_messages[0], modified_signature)
        self.assertFalse(is_valid, "修改后的签名应该验证失败")
    
    def test_performance(self):
        """
        简单的性能测试
        """
        import time
        
        # 创建一个新的签名方案实例
        perf_dgs = DynamicGroupSignature()
        perf_dgs.setup()
        
        # 加入10个成员
        perf_members = [f'perf_member_{i}' for i in range(10)]
        for member_id in perf_members:
            perf_dgs.join_group(member_id)
        
        # 测试签名生成性能
        start_time = time.time()
        for member_id in perf_members:
            perf_dgs.sign(self.test_messages[0], member_id)
        sign_time = time.time() - start_time
        
        # 测试签名验证性能
        signatures = []
        for member_id in perf_members:
            signatures.append(perf_dgs.sign(self.test_messages[0], member_id))
        
        start_time = time.time()
        for signature in signatures:
            perf_dgs.verify(self.test_messages[0], signature)
        verify_time = time.time() - start_time
        
        info(f"\n性能测试结果:")
        info(f"10个签名生成时间: {sign_time:.4f}秒")
        info(f"10个签名验证时间: {verify_time:.4f}秒")
    

if __name__ == '__main__':
    # 运行所有测试
    unittest.main()