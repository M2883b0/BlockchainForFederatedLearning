"""
  - 区块链联邦学习系统 -
	客户端脚本
	
	本模块实现了联邦学习系统中的客户端功能，包括：
	1. 与矿工节点通信获取全局模型
	2. 在本地训练模型并生成更新
	3. 将更新发送到区块链网络
	4. 保存本地模型副本
"""

from federatedlearner import *
from blockchain import *
from uuid import uuid4
import requests
import data.federated_data_extractor as dataext
import time

class Client:
	"""
	客户端类
	
	实现联邦学习客户端的核心功能，包括模型获取、本地训练和更新提交
	"""
	def __init__(self, miner, dataset):
		"""
		初始化客户端
		
		参数:
			miner: 矿工节点地址
			dataset: 数据集路径
		"""
		self.id = str(uuid4()).replace('-','')
		self.miner = miner
		self.dataset = self.load_dataset(dataset)

	def get_last_block(self):
		"""
		获取区块链中的最后一个区块
		
		返回:
			最后一个区块的哈希信息
		"""
		return self.get_chain()[-1]

	def get_chain(self):
		"""
		获取完整的区块链信息
		
		返回:
			区块链的哈希链
		"""
		response = requests.get('http://{node}/chain'.format(node=self.miner))
		if response.status_code == 200:
			return response.json()['chain']

	def get_full_block(self, hblock):
		"""
		获取完整的区块数据
		
		参数:
			hblock: 区块哈希信息
			
		返回:
			Block对象
		"""
		response = requests.post('http://{node}/block'.format(node=self.miner),
			json={'hblock': hblock})
		if response.json()['valid']:
			return Block.from_string(response.json()['block'])
		print("Invalid block!")
		return None

	def get_model(self, hblock):
		"""
		从区块中获取模型数据
		
		参数:
			hblock: 区块哈希信息
			
		返回:
			模型参数字典
		"""
		response = requests.post('http://{node}/model'.format(node=self.miner),
			json={'hblock': hblock})
		if response.json()['valid']:
			return dict(pickle.loads(codecs.decode(response.json()['model'].encode(), "base64")))
		print("Invalid model!")
		return None

	def get_miner_status(self):
		"""
		获取矿工节点的状态
		
		返回:
			矿工状态信息
		"""
		response = requests.get('http://{node}/status'.format(node=self.miner))
		if response.status_code == 200:
			return response.json()

	def load_dataset(self, name):
		"""
		加载客户端训练数据集
		
		参数:
			name: 数据集路径
			
		返回:
			数据集对象
		"""
		if name==None:
			return None
		return dataext.load_data(name)

	def update_model(self, model, steps):
		"""
		在客户端本地训练模型并计算梯度
		
		参数:
			model: 基础模型参数
			steps: 训练步数
			
		返回:
			gradient: 梯度（训练后模型 - 基础模型）
			accuracy: 模型准确率
			训练时间
		"""
		reset()
		t = time.time()
		worker = NNWorker(self.dataset['train_images'],
			self.dataset['train_labels'],
			self.dataset['test_images'],
			self.dataset['test_labels'],
			len(self.dataset['train_images']),
			self.id,
			steps)
		worker.build(model)
		worker.train()
		updated_model = worker.get_model()
		# 计算梯度：训练后模型 - 基础模型
		gradient = {}
		for key in model:
			gradient[key] = updated_model[key] - model[key]
		accuracy = worker.evaluate()
		worker.close()
		return gradient,accuracy,time.time()-t

	def send_update(self, gradient, cmp_time, baseindex):
		"""
		将客户端梯度发送到区块链
		
		参数:
			gradient: 梯度（训练后模型 - 基础模型）
			cmp_time: 计算时间
			baseindex: 基础模型索引
		"""
	
		requests.post('http://{node}/transactions/new'.format(node=self.miner),
			json = {
				'client': self.id,
				'baseindex': baseindex,
				'update': codecs.encode(pickle.dumps(sorted(gradient.items())), "base64").decode(),
				'datasize': len(self.dataset['train_images']),
				'computing_time': cmp_time
			})

	def work(self, device_id, elimit):
		"""
		客户端工作主循环
		
		检查挖矿状态，获取全局模型，进行本地训练并发送更新
		
		参数:
			device_id: 设备标识
			elimit: 训练轮次限制
		"""
		last_model = -1

		for i in range(elimit):
			wait = True
			while wait:
				status = client.get_miner_status()
				if status['status']!="receiving" or last_model==status['last_model_index']:
					time.sleep(10)
					print("waiting")
				else:
					wait = False
			hblock = client.get_last_block()
			baseindex = hblock['index']
			print("Accuracy global model",hblock['accuracy'])
			last_model = baseindex
			model = client.get_model(hblock)
			gradient,accuracy,cmp_time = client.update_model(model,10)
			with open("clients/device"+str(device_id)+"_model_v"+str(i)+".block","wb+") as f:
				pickle.dump(gradient,f)
			#j = j+
			print("Accuracy local update---------"+str(device_id)+"--------------:",accuracy)
			client.send_update(gradient,cmp_time,baseindex)


if __name__ == '__main__':

	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('-m', '--miner', default='127.0.0.1:5000', help='Address of miner')
	parser.add_argument('-d', '--dataset', default='data/mnist.d', help='Path to dataset')
	parser.add_argument('-e', '--epochs', default=10,type=int, help='Number of epochs')
	args = parser.parse_args()
	client = Client(args.miner,args.dataset)
	print("--------------")
	print(client.id," Dataset info:")
	# Data_size, Number_of_classes = dataext.get_dataset_details(client.dataset)
	dataext.get_dataset_details(client.dataset)
	print("--------------")
	device_id = client.id[:2]
	print(device_id,"device_id")
	print("--------------")
	client.work(device_id, args.epochs)
