import pandas as pd
import numpy as np

from utils import TRAIN_DATA, DEV_DATA, TEST_DATA

class BucketedDataIterator():
    def __init__(self, df, num_buckets=10):
        df = df.sort_values('length').reset_index(drop=True)
        self.size = len(df) / num_buckets
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.ix[bucket*self.size: (bucket + 1)*self.size - 1])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, batch_size):
        if np.any(self.cursor + batch_size + 1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0, self.num_buckets)

        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i] + batch_size - 1]

        words = map(lambda x: map(int, x.split(",")), res['words'].tolist())
        tags = map(lambda x: map(int, x.split(",")), res['tags'].tolist())

        self.cursor[i] += batch_size

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([batch_size, maxlen], dtype=np.int32)
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = words[i]
        for i, y_i in enumerate(y):
            y_i[:res['length'].values[i]] = tags[i]

        return x, y, res['length'].values
