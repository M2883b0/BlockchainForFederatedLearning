"""
- 区块链联邦学习系统 -
  区块模型解析工具
  
  本脚本用于测试和验证区块链中存储的模型参数，包括：
  1. 从区块文件中加载序列化的区块数据
  2. 提取其中的模型参数
  3. 将参数以可读形式写入文本文件
"""

import pickle

# 注意：需要将路径修改为实际的文件路径
# 加载区块文件
try:
    with open("/home/user/Documents/RRBFL/blocks/federated_model1.block", "rb") as f:
        # 反序列化区块对象
        block = pickle.load(f)
        # 提取区块中的基础模型参数
        model = block.basemodel
        
        # 将模型参数写入文本文件
        with open("/home/user/Documents/RRBFL/blocks/federated_model1.txt", "a") as f_out:
            for keys, values in sorted(model.items()):
                # 写入参数名和参数值
                f_out.write(str(keys) + ' ->>> ' + str(values) + '\n\n')
            # 文件会在with块结束时自动关闭，不需要显式调用close()
    print("模型参数已成功解析并写入文本文件")
except Exception as e:
    print(f"处理过程中出现错误: {e}")
    print("请检查文件路径是否正确，并确保区块文件存在")



