"""
 - 区块链联邦学习系统 -
   区块链脚本实现
   
   本模块实现了基于区块链的联邦学习核心功能，包括：
   1. 全局模型聚合计算
   2. 区块链数据结构和共识机制
   3. 区块生成、验证和存储
   4. 更新管理和冲突解决
"""

import codecs
import hashlib
import json
import random
import time
from urllib.parse import urlparse

import requests

import data.federated_data_extractor as dataext
from federatedlearner import *


def compute_global_model(base, updates, lrate):
    """
    计算全局模型的函数
    
    根据每轮接收到的客户端更新计算全局模型
    
    参数:
        base: 基础模型参数
        updates: 客户端更新字典
        lrate: 学习率
    
    返回:
        accuracy: 全局模型的准确率
        upd: 更新后的全局模型参数
    """

    upd = dict()
    for x in ['w1','w2','wo','b1','b2','bo']:
        upd[x] = np.array(base[x], copy=True)
    number_of_clients = len(updates)
    # 客户端上传的是训练后的完整模型，不是梯度
    # 正确的联邦平均步骤：
    # 1. 计算每个客户端模型相对于基础模型的梯度（差）
    # 2. 对所有客户端的梯度进行加权平均
    # 3. 将平均梯度应用到基础模型上
    for client in updates.keys():
        for x in ['w1','w2','wo','b1','b2','bo']:
            model = updates[client].update
            upd[x] += (lrate/number_of_clients)*(model[x]+base[x])
    upd["size"] = 0
    reset()
    dataset = dataext.load_data("data/mnist.d")
    worker = NNWorker(None,
        None,
        dataset['test_images'],
        dataset['test_labels'],
        0,
        "validation")
    worker.build(upd)
    accuracy = worker.evaluate()
    worker.close()
    return accuracy,upd

def find_len(text, strk):
    """
    在文本中查找指定字符串并返回其起始位置和长度
    
    参数:
        text: 要搜索的文本
        strk: 要查找的字符串
    
    返回:
        起始位置和字符串长度
    """
    return text.find(strk),len(strk)

class Update:
    """
    更新类
    
    用于存储和管理客户端提交的模型更新信息
    """
    def __init__(self, client, baseindex, update, datasize, computing_time, timestamp=time.time()):
        """
        初始化更新参数
        
        参数:
            client: 客户端标识
            baseindex: 基础模型索引
            update: 更新的模型参数
            datasize: 数据大小
            computing_time: 计算时间
            timestamp: 时间戳
        """
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time

    @staticmethod
    def from_string(metadata):
        """
        从字符串中解析更新信息
        
        参数:
            metadata: 包含更新信息的字符串
            
        返回:
            Update对象
        """
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'baseindex':")
        i3,l3 = find_len(metadata,"'update': ")
        i4,l4 = find_len(metadata,"'client':")
        i5,l5 = find_len(metadata,"'datasize':")
        i6,l6 = find_len(metadata,"'computing_time':")
        baseindex = int(metadata[i2+l2:i3].replace(",",'').replace(" ",""))
        update = dict(pickle.loads(codecs.decode(metadata[i3+l3:i4-1].encode(), "base64")))
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        client = metadata[i4+l4:i5].replace(",",'').replace(" ","")
        datasize = int(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        computing_time = float(metadata[i6+l6:].replace(",",'').replace(" ",""))
        return Update(client,baseindex,update,datasize,computing_time,timestamp)


    def __str__(self):
        """
        将更新对象转换为字符串表示
        
        返回:
            格式化的更新字符串
        """
        return "'timestamp': {timestamp},\
            'baseindex': {baseindex},\
            'update': {update},\
            'client': {client},\
            'datasize': {datasize},\
            'computing_time': {computing_time}".format(
                timestamp = self.timestamp,
                baseindex = self.baseindex,
                update = codecs.encode(pickle.dumps(sorted(self.update.items())), "base64").decode(),
                client = self.client,
                datasize = self.datasize,
                computing_time = self.computing_time
            )


class Block:
    """
    区块类
    
    用于存储区块链中的区块数据
    """
    def __init__(self, miner, index, basemodel, accuracy, updates, timestamp=time.time()):
        """
        初始化区块参数
        
        参数:
            miner: 矿工标识
            index: 区块索引
            basemodel: 基础模型
            accuracy: 模型准确率
            updates: 客户端更新集合
            timestamp: 时间戳
        """
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(metadata):
        """
        从字符串中解析区块信息
        
        参数:
            metadata: 包含区块信息的字符串
            
        返回:
            Block对象
        """
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'basemodel': ")
        i3,l3 = find_len(metadata,"'index':")
        i4,l4 = find_len(metadata,"'miner':")
        i5,l5 = find_len(metadata,"'accuracy':")
        i6,l6 = find_len(metadata,"'updates':")
        i9,l9 = find_len(metadata,"'updates_size':")
        index = int(metadata[i3+l3:i4].replace(",",'').replace(" ",""))
        miner = metadata[i4+l4:i].replace(",",'').replace(" ","")
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        basemodel = dict(pickle.loads(codecs.decode(metadata[i2+l2:i5-1].encode(), "base64")))
        accuracy = float(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        su = metadata[i6+l6:i9]
        su = su[:su.rfind("]")+1]
        updates = dict()
        for x in json.loads(su):
            isep,lsep = find_len(x,"@|!|@")
            updates[x[:isep]] = Update.from_string(x[isep+lsep:])
        updates_size = int(metadata[i9+l9:].replace(",",'').replace(" ",""))
        return Block(miner,index,basemodel,accuracy,updates,timestamp)

    def __str__(self):
        """
        将区块对象转换为字符串表示
        
        返回:
            格式化的区块字符串
        """
        return "'index': {index},\
            'miner': {miner},\
            'timestamp': {timestamp},\
            'basemodel': {basemodel},\
            'accuracy': {accuracy},\
            'updates': {updates},\
            'updates_size': {updates_size}".format(
                index = self.index,
                miner = self.miner,
                basemodel = codecs.encode(pickle.dumps(sorted(self.basemodel.items())), "base64").decode(),
                accuracy = self.accuracy,
                timestamp = self.timestamp,
                updates = str([str(x[0])+"@|!|@"+str(x[1]) for x in sorted(self.updates.items())]),
                updates_size = str(len(self.updates))
            )



class Blockchain(object):
    """
    区块链类
    
    实现区块链的核心功能，包括区块生成、验证、存储等
    """
    def __init__(self, miner_id, base_model=None, gen=False, update_limit=10, time_limit=1800):
        """
        初始化区块链
        
        参数:
            miner_id: 矿工标识
            base_model: 基础模型
            gen: 是否生成创世区块
            update_limit: 更新数量限制
            time_limit: 时间限制
        """
        super(Blockchain,self).__init__()
        self.miner_id = miner_id
        self.curblock = None
        self.hashchain = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit
        
        if gen:
            genesis,hgenesis = self.make_block(base_model=base_model,previous_hash=1)
            self.store_block(genesis,hgenesis)
        self.nodes = set()

    def register_node(self, address):
        """
        注册新节点
        
        参数:
            address: 节点地址
        """
        if address[:4] != "http":
            address = "http://"+address
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        print("Registered node",address)

    def make_block(self, previous_hash=None, base_model=None):
        """
        创建新区块
        
        参数:
            previous_hash: 前一个区块的哈希值
            base_model: 基础模型
            
        返回:
            新区块和区块哈希信息
        """
        accuracy = 0
        basemodel = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain)>0:
            update_limit = self.last_block['update_limit']
            time_limit = self.last_block['time_limit']
        if previous_hash==None:
            previous_hash = self.hash(str(sorted(self.last_block.items())))
        if base_model!=None:
            accuracy = base_model['accuracy']
            basemodel = base_model['model']
        elif len(self.current_updates)>0:
            base = self.curblock.basemodel
            accuracy,basemodel = compute_global_model(base,self.current_updates,1)
        index = len(self.hashchain)+1
        block = Block(
            miner = self.miner_id,
            index = index,
            basemodel = basemodel,
            accuracy = accuracy,
            updates = self.current_updates
            )
        hashblock = {
            'index':index,
            'hash': self.hash(str(block)),
            'proof': random.randint(0,100000000),
            'previous_hash': previous_hash,
            'miner': self.miner_id,
            'accuracy': str(accuracy),
            'timestamp': time.time(),
            'time_limit': time_limit,
            'update_limit': update_limit,
            'model_hash': self.hash(codecs.encode(pickle.dumps(sorted(block.basemodel.items())), "base64").decode())
            }
        return block,hashblock

    def store_block(self, block, hashblock):
        """
        存储区块
        
        将当前区块序列化并存储到文件，然后更新区块链状态
        
        参数:
            block: 区块对象
            hashblock: 区块哈希信息
            
        返回:
            区块哈希信息
        """
        if self.curblock:
            with open("blocks/federated_model"+str(self.curblock.index)+".block","wb") as f:
                pickle.dump(self.curblock,f)
        self.curblock = block
        self.hashchain.append(hashblock)
        self.current_updates = dict()
        return hashblock

    def new_update(self, client, baseindex, update, datasize, computing_time):
        """
        添加新的客户端更新
        
        参数:
            client: 客户端标识
            baseindex: 基础模型索引
            update: 更新的模型参数
            datasize: 数据大小
            computing_time: 计算时间
            
        返回:
            下一个区块的索引
        """
        self.current_updates[client] = Update(
            client = client,
            baseindex = baseindex,
            update = update,
            datasize = datasize,
            computing_time = computing_time
            )
        return self.last_block['index']+1

    @staticmethod
    def hash(text):
        """
        计算文本的SHA-256哈希值
        
        参数:
            text: 要哈希的文本
            
        返回:
            哈希值
        """
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def last_block(self):
        """
        获取最后一个区块
        
        返回:
            最后一个区块的哈希信息
        """
        return self.hashchain[-1]


    def proof_of_work(self, stop_event):
        """
        工作量证明算法
        
        寻找满足难度要求的随机数
        
        参数:
            stop_event: 停止事件
            
        返回:
            区块哈希信息和是否停止的标志
        """
        block,hblock = self.make_block()
        stopped = False
        while self.valid_proof(str(sorted(hblock.items()))) is False:
            if stop_event.is_set():
                stopped = True
                break
            hblock['proof'] += 1
            if hblock['proof']%1000==0:
                print("mining",hblock['proof'])
        if stopped==False:
            self.store_block(block,hblock)
        if stopped:
            print("Stopped")
        else:
            print("Done")
        return hblock,stopped

    @staticmethod
    def valid_proof(block_data):
        """
        验证工作量证明
        
        检查哈希值是否满足难度要求
        
        参数:
            block_data: 区块数据
            
        返回:
            是否有效
        """
        guess_hash = hashlib.sha256(block_data.encode()).hexdigest()
        k = "00000"
        return guess_hash[:len(k)] == k


    def valid_chain(self, hchain):
        """
        验证区块链的有效性
        
        检查区块链中的每个区块是否有效
        
        参数:
            hchain: 区块链哈希链
            
        返回:
            是否有效
        """
        last_block = hchain[0]
        curren_index = 1
        while curren_index<len(hchain):
            hblock = hchain[curren_index]
            if hblock['previous_hash'] != self.hash(str(sorted(last_block.items()))):
                print("prev_hash diverso",curren_index)
                return False
            if not self.valid_proof(str(sorted(hblock.items()))):
                print("invalid proof",curren_index)
                return False
            last_block = hblock
            curren_index += 1
        return True

    def resolve_conflicts(self, stop_event):
        """
        解决区块链冲突
        
        从邻居节点获取最长有效链并更新本地链
        
        参数:
            stop_event: 停止事件
            
        返回:
            是否解决了冲突
        """
        neighbours = self.nodes
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)
        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length>max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node
        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            resp = requests.post('http://{node}/block'.format(node=bnode),
                json={'hblock': hblock})
            self.current_updates = dict()
            if resp.status_code == 200:
                if resp.json()['valid']:
                    self.curblock = Block.from_string(resp.json()['block'])
            return True
        return False
