"""
 - 区块链联邦学习系统 -
           矿工节点脚本
           
           本模块实现了区块链联邦学习系统中的矿工节点功能，包括：
           1. 区块生成和挖矿过程管理
           2. 客户端更新接收和处理
           3. 区块链网络通信和共识
           4. 模型存储和检索API
"""

import glob
import os
from threading import Thread, Event
from uuid import uuid4

from flask import jsonify, request, Flask

from blockchain import *
from federatedlearner import *


def make_base():
    """
    为创世区块创建基础模型
    
    在第一组客户端数据上进行基础级别训练，生成创世区块所需的初始模型
    
    返回:
        包含模型参数和精度的字典
    """
    reset()
    with open("data/federated_data_0.d",'rb') as f:
        dataset = pickle.load(f)
    worker = NNWorker(dataset["train_images"],
        dataset["train_labels"],
        dataset["test_images"],
        dataset["test_labels"],
        0,
        "base0")
    worker.build_base()
    model = dict()
    model['model'] = worker.get_model()
    model['accuracy'] = worker.evaluate()
    worker.close()
    return model


class PoWThread(Thread):
    """
    工作量证明线程类
    
    在独立线程中执行工作量证明计算
    """
    def __init__(self, stop_event, blockchain, node_identifier):
        """
        初始化工作量证明线程
        
        参数:
            stop_event: 停止事件标志
            blockchain: 区块链对象
            node_identifier: 节点标识符
        """
        self.stop_event = stop_event
        Thread.__init__(self)
        self.blockchain = blockchain
        self.node_identifier = node_identifier
        self.response = None

    def run(self):
        """
        线程运行函数
        
        执行工作量证明算法，并在完成时调用回调函数
        """
        block,stopped = self.blockchain.proof_of_work(self.stop_event)
        self.response = {
            'message':"End mining",
            'stopped': stopped,
            'block': str(block)
        }
        on_end_mining(stopped)


STOP_EVENT = Event()

app = Flask(__name__)
status = {
    's':"receiving",
    'id':str(uuid4()).replace('-',''),
    'blockchain': None,
    'address' : ""
    }

def mine():
    """
    启动挖矿过程
    
    清除停止事件并创建新的工作量证明线程开始挖矿
    """
    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT,status["blockchain"],status["id"])
    status['s'] = "mining"
    thread.start()

def on_end_mining(stopped):
    """
    挖矿结束回调函数
    
    参数:
        stopped: 是否因为停止事件而结束挖矿
    """
    if status['s'] == "receiving":
        return
    if stopped:
        status["blockchain"].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status["blockchain"].nodes:
        requests.get('http://{node}/stopmining'.format(node=node))

@app.route('/transactions/new',methods=['POST'])
def new_transaction():
    """
    接收新的客户端更新
    
    处理客户端提交的模型更新，并在满足条件时触发挖矿
    """
    if status['s'] != "receiving":
        return 'Miner not receiving', 400
    values = request.get_json()

    required = ['client','baseindex','update','datasize','computing_time']
    if not all(k in values for k in required):
        return 'Missing values', 400
    if values['client'] in status['blockchain'].current_updates:
        return 'Model already stored', 400
    index = status['blockchain'].new_update(values['client'],
        values['baseindex'],
        dict(pickle.loads(codecs.decode(values['update'].encode(), "base64"))),
        values['datasize'],
        values['computing_time'])
    for node in status["blockchain"].nodes:
        requests.post('http://{node}/transactions/new'.format(node=node),
            json=request.get_json())
    if (status['s']=='receiving' and (
        len(status["blockchain"].current_updates)>=status['blockchain'].last_block['update_limit']
        or time.time()-status['blockchain'].last_block['timestamp']>status['blockchain'].last_block['time_limit'])):
        mine()
    response = {'message': "Update will be added to block {index}".format(index=index)}
    return jsonify(response),201

@app.route('/status',methods=['GET'])
def get_status():
    """
    获取矿工节点状态
    
    返回当前挖矿状态和最后一个模型的索引
    """
    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].last_block['index']
        }
    return jsonify(response),200

@app.route('/chain',methods=['GET'])
def full_chain():
    """
    获取完整的区块链信息
    
    返回整个哈希链和其长度
    """
    response = {
        'chain': status['blockchain'].hashchain,
        'length':len(status['blockchain'].hashchain)
    }
    return jsonify(response),200

@app.route('/nodes/register',methods=['POST'])
def register_nodes():
    """
    注册新节点
    
    将新的矿工节点添加到区块链网络中
    """
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Enter valid nodes in the list ", 400
    for node in nodes:
        if node!=status['address'] and not node in status['blockchain'].nodes:
            status['blockchain'].register_node(node)
            for miner in status['blockchain'].nodes:
                if miner!=node:
                    print("node",node,"miner",miner)
                    requests.post('http://{miner}/nodes/register'.format(miner=miner),
                        json={'nodes': [node]})
    response = {
        'message':"New nodes have been added",
        'total_nodes':list(status['blockchain'].nodes)
    }
    return jsonify(response),201

@app.route('/block',methods=['POST'])
def get_block():
    """
    获取指定区块
    
    根据区块哈希信息返回完整的区块数据
    """
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block","rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
            json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block","wb") as f:
                    pickle.dump(block,f)
    valid = False
    if Blockchain.hash(str(block))==hblock['hash']:
        valid = True
    response = {
        'block': str(block),
        'valid': valid
    }
    return jsonify(response),200

@app.route('/model',methods=['POST'])
def get_model():
    """
    获取区块中的模型
    
    根据区块哈希信息返回其中的模型参数
    """
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block","rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
            json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block","wb") as f:
                    pickle.dump(block,f)
    valid = False
    model = block.basemodel
    if Blockchain.hash(codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode())==hblock['model_hash']:
        valid = True
    response = {
        'model': codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode(),
        'valid': valid
    }
    return jsonify(response),200

@app.route('/nodes/resolve',methods=["GET"])
def consensus():
    """
    解决区块链冲突
    
    执行共识算法，确保所有节点使用相同的区块链
    """
    replaced = status['blockchain'].resolve_conflicts(STOP_EVENT)
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': status['blockchain'].hashchain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': status['blockchain'].hashchain
        }
    return jsonify(response), 200


@app.route('/stopmining',methods=['GET'])
def stop_mining():
    """
    停止挖矿过程
    
    触发区块链冲突解决并停止当前挖矿
    """
    status['blockchain'].resolve_conflicts(STOP_EVENT)
    response = {
        'mex':"stopped!"
    }
    return jsonify(response),200

def delete_prev_blocks():
    """
    删除之前的区块文件
    
    清理blocks目录下的所有区块文件
    """
    files = glob.glob('blocks/*.block')
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    parser.add_argument('-i', '--host', default='127.0.0.1', help='IP address of this miner')
    parser.add_argument('-g', '--genesis', default=0, type=int, help='instantiate genesis block')
    parser.add_argument('-l', '--ulimit', default=10, type=int, help='number of updates stored in one block')
    parser.add_argument('-ma', '--maddress', help='other miner IP:port')
    args = parser.parse_args()
    address = "{host}:{port}".format(host=args.host,port=args.port)
    status['address'] = address
    if args.genesis==0 and args.maddress==None:
        raise ValueError("Must set genesis=1 or specify maddress")
    delete_prev_blocks()
    if args.genesis==1:
        model = make_base()
        print("base model accuracy:",model['accuracy'])
        status['blockchain'] = Blockchain(address,model,True,args.ulimit)
    else:
        status['blockchain'] = Blockchain(address)
        status['blockchain'].register_node(args.maddress)
        requests.post('http://{node}/nodes/register'.format(node=args.maddress),
            json={'nodes': [address]})
        status['blockchain'].resolve_conflicts(STOP_EVENT)
    app.run(host=args.host,port=args.port)
