from __future__ import absolute_import
from __future__ import print_function

import os
import pickle as pkl
import numpy as np
import sys

DATA_PATH = './data/'

# np.random.seed(123)


class GDataset(object):

    def __init__(self, name, directory):

        pass

        self.name = name
        self.data = {}
        self.trigger = {}
        self.entity = {}
        self.node = {}

        if dir:
            self.path = os.path.join(directory, self.name)
        # print(self.path)
        self.n_triggers = -1
        self.n_entities = -1
        self.n_sents = -1
        self.n_nodes = -1

    def load(self):
        pass

        node_path = os.path.join(self.path, 'node.txt')
        # self.n_event = 0

        # raw->元组
        splits = ["train", "test", "valid"]
        for split in splits:
            data_path = os.path.join(self.path, split+'.txt')
            self.data[split] = np.array(self.read_examples(data_path))

    def get_shape(self):
        return self.n_nodes, self.n_triggers, self.n_sents

    def get_nodes_num(self):
        return self.n_nodes

    def get_triggers_nums(self):
        return self.n_triggers

    def read_examples(self, dataset_file):
        """

        :param dataset_file:
        :return: np.array(int64)
        """
        # print("path:{}", dataset_file)
        dataset_file = dataset_file.replace('\\', '/')
        examples = []
        with open(dataset_file, "r") as lines:
            for line in lines:
                line = line.strip().split('\t')
                pass
                s, r, o, st = line[:4]
                examples.append([s, r, o])
        return np.array(examples).astype("int64")

    def read_dictionary(self, filename):
        """

        :param filename:
        :return: dict {id(int): str}
        """
        # print("path:{}", filename)
        d = {}
        filename = filename.replace('\\', '/')
        with open(filename, "r", encoding='utf-8') as lines:
            for line in lines:
                line = line.strip().split('\t')
                d[int(line[1])] = line[0]
        return d


def load_dataset(dataset):
    """

    :param dataset:
    :return: class KGDataset
    """
    if dataset in ['ECB+']:
        return GDataset(dataset, DATA_PATH)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
