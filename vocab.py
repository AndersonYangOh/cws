# -*- coding: utf-8 -*-

import os
import numpy as np
from collections import defaultdict

from utils import WORD_VEC

class Vocab(object):
    def __init__(self, path):
        self.path = path
        self.word2idx = defaultdict(int)
        self.word_vectors = None
        self.load_data()

    def load_data(self):
        with open(self.path, 'r') as f:
            line = f.readline().strip().split(" ")
            N, dim = map(int, line)

            self.word_vectors = []
            mean_vector = np.zeros(dim)

            idx = 0
            for k in range(N):
                line = f.readline().strip().split(" ")
                self.word2idx[line[0]] = idx
                vector = np.asarray(map(float, line[1:]), dtype=np.float32)
                self.word_vectors.append(vector)
                idx += 1
                mean_vector += vector

            # unkown word
            self.word2idx['UNK'] = idx
            mean_vector /= N
            self.word_vectors.append(mean_vector)

            self.word_vectors = np.asarray(self.word_vectors, dtype=np.float32)


class Tag(object):
    def __init__(self):
        self.tag2idx = defaultdict(int)
        self.define_tags()

    def define_tags(self):
        self.tag2idx['B'] = 0
        self.tag2idx['I'] = 1
        self.tag2idx['E'] = 2
        self.tag2idx['S'] = 3
