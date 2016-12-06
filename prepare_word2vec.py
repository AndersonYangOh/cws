# -*- coding: utf-8 -*-

import os

import text_utils
from utils import LABELED_DATA, WORD2VEC_INPUT

def process_file(path, output):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sents = text_utils.clean_and_separate(line)
            for sent in sents:
                output.write(sent + "\n")

def main():
    output = open(WORD2VEC_INPUT, 'w')
    for dirpath, dirnames, filenames in os.walk(LABELED_DATA):
        for filename in filenames:
            if filename.endswith('.txt'):
                path = os.path.join(dirpath, filename)
                print path
                process_file(path, output)

if __name__ == "__main__":
    main()