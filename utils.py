# -*- coding: utf-8 -*-

import os

DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, 'data')
MODEL_DIR = os.path.join(DIR, 'models')

LABELED_DATA = os.path.join(DATADIR, '2014')

WORD2VEC_INPUT = os.path.join(DATADIR, 'word2vec_input.txt')

WORD_VEC = os.path.join(MODEL_DIR, 'vec.txt')

ALL_DATA = os.path.join(DATADIR, 'all.csv')

TRAIN_DATA = os.path.join(DATADIR, 'train.csv')
DEV_DATA = os.path.join(DATADIR, 'dev.csv')
TEST_DATA = os.path.join(DATADIR, 'test.csv')

DEV_SIZE = 10000
TEST_SIZE = 10000

MAX_LEN = 80