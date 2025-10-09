# -*- coding:utf-8 -*-
# @FileName :demo_dgs.py
# @Time :2025/10/9 14:33
# @Author :M2883b0

"""
- 基于格密码的动态群签名库 -
  演示脚本

  本脚本演示了如何使用基于格密码的动态群签名库，包括：
  1. 初始化群
  2. 添加群成员
  3. 生成和验证群签名
  4. 撤销群成员
"""

from ..src.DGS import DynamicGroupSignature, LatticeParameters


def main():
    """
    主函数，演示动态群签名库的使用
    """
    print("===== 基于格密码的动态群签名库演示 =====")
    
    # 创建签名方案实例
    dgs = DynamicGroupSignature()
    
    # 初始化群
    print("1. 初始化群...")
    public_key = dgs.setup()
    print(f"   群已初始化，群ID: {public_key['group_id']}")
    
    # 成员加入群
    print("2. 成员加入群...")
    members = ['member1', 'member2', 'member3']
    for member in members:
        dgs.join_group(member)
        print(f"   成员 {member} 已加入群")
    
    # 生成签名
    print("3. 生成群签名...")
    message = "这是一个用于测试的消息".encode('utf-8')
    signature = dgs.sign(message, 'member1')
    print(f"   成员1 已为消息生成签名")
    
    # 验证签名
    print("4. 验证群签名...")
    is_valid = dgs.verify(message, signature)
    print(f"   签名验证结果: {'有效' if is_valid else '无效'}")
    
    # 撤销成员
    print("5. 撤销群成员...")
    dgs.revoke_member('member2')
    print(f"   成员2 已被撤销")
    
    # 尝试使用已撤销成员的ID签名
    print("6. 尝试使用已撤销成员的ID签名...")
    try:
        invalid_signature = dgs.sign(message, 'member2')
        print("   错误: 应该无法使用已撤销成员的ID签名")
    except ValueError as e:
        print(f"   预期行为: {e}")
    
    # 验证群信息
    print("7. 查看群信息...")
    group_info = dgs.get_group_info()
    print(f"   群信息: {group_info}")
    
    # 导出和导入公钥
    print("8. 导出和导入公钥...")
    pk_bytes = dgs.export_public_key()
    print(f"   公钥已导出，大小: {len(pk_bytes)} 字节")
    
    # 创建新实例并导入公钥
    new_dgs = DynamicGroupSignature()
    new_dgs.import_public_key(pk_bytes)
    print("   公钥已成功导入到新实例")
    
    print("===== 演示结束 =====")


if __name__ == '__main__':
    main()