"""
- 区块链联邦学习系统 -
  CSV文件生成脚本
  
  本脚本用于生成记录客户端设备信息的CSV文件，包含：
  1. 扫描clients目录下的所有模型文件
  2. 提取设备ID信息
  3. 生成包含设备ID、文件路径和模型版本的CSV文件
"""

import glob
import csv

# 设置参数
path = 'clients/'  # 客户端模型文件目录
n_devices = 2      # 设备数量
Model_version = 0  # 模型版本
l = len(path)      # 路径长度，用于后续字符串处理
i = 1              # 当前处理的设备索引

# 扫描客户端目录下所有的.block文件
Device_gradient_path = glob.glob(path+"*.block")
print(Device_gradient_path)

# 创建字典存储客户端详细信息
client_details = dict()

# 提取每个模型文件对应的设备ID
for gradient in Device_gradient_path:
    # 从文件路径中提取设备ID（第6到第7个字符）
    client_details[gradient] = gradient[l+5 : l+7]

# 初始化CSV数据结构
csvData = [[] for _ in range(n_devices+1)]
csvData[0] = ['Device_id', 'Device_delta_path', 'Model_version']  # 设置表头

# 填充CSV数据
for k, v in client_details.items():
    csvData[i].append(v)       # 添加设备ID
    csvData[i].append(k)       # 添加文件路径
    csvData[i].append(Model_version)  # 添加模型版本
    i += 1

# 生成CSV文件
with open('DeltaOffChainDatabase.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()
    
