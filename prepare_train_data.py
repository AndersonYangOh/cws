# -*- coding: utf-8 -*-

import pandas as pd
import random
import os
import csv

import text_utils
from utils import LABELED_DATA, ALL_DATA, MAX_LEN, WORD_VEC
from utils import TRAIN_DATA, DEV_DATA, TEST_DATA, DEV_SIZE, TEST_SIZE
from vocab import Vocab, Tag

VOCABS = Vocab(WORD_VEC)
TAGS = Tag()

def to_index(words, tags):
    word_idx = []
    for word in words:
        word = word.encode("utf8")
        if word in VOCABS.word2idx:
            word_idx.append(VOCABS.word2idx[word])
        else:
            word_idx.append(VOCABS.word2idx['UNK'])

    tag_idx = [TAGS.tag2idx[tag] for tag in tags]

    return ','.join(map(str, word_idx)), ','.join(map(str, tag_idx))

def process_file(path, output):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sents = text_utils.clean_tags(line)
            for words, tags in sents:
                words = words[:MAX_LEN]
                tags = tags[:MAX_LEN]
                word_idx, tag_idx = to_index(words, tags)
                length = len(words)
                output.writerow([word_idx, tag_idx, length])    

def process_all_data():
    f = open(ALL_DATA, 'w')
    output = csv.writer(f)
    output.writerow(['words', 'tags', 'length'])

    for dirpath, dirnames, filenames in os.walk(LABELED_DATA):
        for filename in filenames:
            if filename.endswith('.txt'):
                path = os.path.join(dirpath, filename)
                print path
                process_file(path, output)


def split_train_dev_test():
    df = pd.read_csv(ALL_DATA)

    # drop sentences shorter than 3
    df = df[df['length'] > 2]

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # select dev, test and train data
    df_dev = df[:DEV_SIZE].reset_index(drop=True)
    df_test = df[-TEST_SIZE:].reset_index(drop=True)
    df_train = df[DEV_SIZE:-TEST_SIZE].reset_index(drop=True)

    # save to csv
    df_train.to_csv(TRAIN_DATA, index=False)
    df_dev.to_csv(DEV_DATA, index=False)
    df_test.to_csv(TEST_DATA, index=False)

if __name__ == "__main__":
    process_all_data()
    split_train_dev_test()