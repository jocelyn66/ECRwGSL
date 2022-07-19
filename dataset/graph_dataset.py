from cmath import sqrt
import itertools
import os
from pydoc import doc
import numpy as np
import json

import numpy as np
from utils.train import add_new_item

DATA_PATH = './data/'

# np.random.seed(123)
# n_events = 3808
# n_entities = 4758

n_events = {'Train': 3808,'Dev': 1245, 'Test': 1780}
n_entities = {'Train': 4758,'Dev': 1476, 'Test': 2055}


class GDataset(object):

    def __init__(self, args):

        self.name = args.dataset
        # self.data = {}  # 
        self.event_idx = {'Train':[], 'Dev':[], 'Test':[]}
        # self.node = {}
        self.event_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}  # train, dev, test
        self.adjacency = {}  # 邻接矩阵
        self.event_coref_adj = {}  # 是event mention(trigger)，且共指关系成立，则为1.
        self.n_nodes = {}
        
        self.rand_node_rate = args.rand_node_rate
        self.n_events = n_events
        self.n_entities = n_entities

        for split in ['Train', 'Dev', 'Test']:
            assert(self.n_events[split] > 0)
            assert(self.n_entities[split] > 0)
            self.n_nodes[split] = self.n_events[split] + self.n_entities[split]

        # self.adjacency['Train'] = self.get_adjacency(args.train_file, 'Train')  # 0. 1.矩阵, 对角线1
        # self.adjacency['Dev'] = self.get_adjacency(args.dev_file, 'Dev')
        # self.adjacency['Test'] = self.get_adjacency(args.test_file, 'Test')
        file = {'Train': args.train_file, 'Dev': args.dev_file,'Test': args.test_file}
        # self.event_coref_adj['Train'] = self.get_event_coref_adj('Train')
        for split in ['Train', 'Dev', 'Test']:
            self.adjacency[split] = self.get_adjacency(file[split], split)
            self.event_coref_adj[split] = self.get_event_coref_adj(split)  # bool矩阵, 对角线无用

    def get_schema(self, path, split=''):
        # chain的schema, item：(chain descrip, id)
        # return: event_schema, entity_schema
        if not split:
            ValueError
        with open(path, 'r') as x:
            schema = json.load(x)
        return schema[1], schema[2]

    def get_adjacency(self, path, split):
        # 构图：
        # 节点：event, entity
        # 边：句子 文档关系
        # 【对角线：1】

        adj = np.zeros((self.n_nodes[split], self.n_nodes[split]))
        last_doc_id = ''
        doc_node_idx = []
        sent_node_idx = []
        cur_idx = -1    # 从0开始顺序处理每个句子，对event chain, entity chain中的mention编号，根据mention出现的顺序
        
        with open(path, 'r') as f:
            lines = f.readlines()

        for _, line in enumerate(lines):

            sent = json.loads(line)
            #  同一文档rand_rate的概率随机放点

            if last_doc_id != sent['doc_id']:
                if doc_node_idx:
                    # rand_rows = np.random.shuffle(doc_node_idx)[:int(num*rand_node_rate)]
                    rows_idx = np.random.rand(len(doc_node_idx)) < sqrt(self.rand_node_rate)
                    cols_idx = np.random.rand(len(doc_node_idx)) < sqrt(self.rand_node_rate)
                    if (rows_idx & cols_idx).any is True:
                        rand_rows = np.array(doc_node_idx)[rows_idx]
                        # rand_rows = doc_node_idx[(np.random.rand(len(doc_node_idx))+1) < (-rand_rate*2)]
                        rand_cols = np.array(doc_node_idx)[cols_idx]
                        adj[rand_rows][rand_cols] = 1

                last_doc_id = sent['doc_id']
                doc_node_idx = []
            
            # event mentions
            for _, event in enumerate(sent['event_coref']):
                cur_idx += 1
                sent_node_idx.append(cur_idx)
                self.event_idx[split].append(cur_idx)
                add_new_item(self.event_chain_dict[split], event['coref_chain'], cur_idx)

            # eneity mentions
            sent_node_idx += [i+cur_idx+1 for i in range(len(sent['entity_coref']))] 
            cur_idx += len(sent['entity_coref'])
            
            # 句子子图
            adj[sent_node_idx[0]:sent_node_idx[-1]+1, sent_node_idx[0]:sent_node_idx[-1]+1] = 1

            doc_node_idx += sent_node_idx
            sent_node_idx = []

        # 处理adj: 对称，对角线0
        adj = np.where((adj + adj.T)>0, 1., 0.)
        adj[np.diag_indices_from(adj)] = 0
        assert(adj.diagonal(offset=0, axis1=0, axis2=1).all()==0)
        return adj
        
    def get_event_node_idx(self, descrip):
        return int(self.schema_event[descrip])

    def get_entity_node_idx(self, descrip):
        return int(self.schema_entity[descrip]) + self.n_events[descrip]

    def get_event_coref_adj(self, split):
        # event coref关系bool矩阵【对角线：1】
        #  for key in event chain dict:
        adj = np.zeros((self.n_nodes[split], self.n_nodes[split]))
        for key in self.event_chain_dict[split]:
            events = self.event_chain_dict[split][key]
            mask = itertools.product(events,events)
            rows, cols = zip(*mask)
            adj[rows, cols] = 1
        # adj = adj + adj.T   # 处理成对称矩阵
        return ((adj + adj.T)>0)[self.event_idx[split], :][:, self.event_idx[split]]

    # def load_adjacency_sp_matrix(dataset, file, n_events, n_entities):
    #     n_nodes = n_events + n_entities
    #     # load train set
    #     # load train schema->event_idx, entity_idx
    #     for line:
    #         # 若doc结束:随机加边
    #         # 句子中的mention list:
    #         # 全排列
    #         # np.stack
        
    #     # npstack -> adj
    #     # 返回对称矩阵
    #     pass
