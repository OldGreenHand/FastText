#!/usr/bin/python

import collections
import math
import os
import random

import numpy as np
import tensorflow as tf
from random import shuffle

import nltk
from nltk import word_tokenize

import sys, getopt
import os

from collections import namedtuple

Dataset = namedtuple('Dataset','sentences labels')

num_classes = 3
learning_rate = 0.05
num_epochs = 2
embedding_dim = 10
label_to_id = {'World':0, 'Entertainment':1, 'Sports':2}
unknown_word_id = 0

def create_label_vec(label):
   # Generate a label vector for a given classification label.
    return label_to_id[label.rstrip()]

def tokenize(sens):
    # Tokenize a given sentence into a sequence of tokens.
    return word_tokenize(sens)

def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id,word) for word in token_seq]

def map_word_to_id(word_to_id, word):
    # map each word to its id.
    return word_to_id[word]

def build_vocab(sens_file_name):
    data = []
    with open(sens_file_name) as f:
        for line in f.readlines():
            tokens = tokenize(line)
            data.extend(tokens)
    count = [['$UNK$', 0]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id

def read_labeled_dataset(sens_file_name, label_file_name, word_to_id):
    sens_file = open(sens_file_name)
    label_file = open(label_file_name)
    data = []
    for label in label_file:
        sens = sens_file.readline()
        word_id_seq = map_token_seq_to_word_id_seq(tokenize(sens), word_to_id)
        data.append((word_id_seq, create_label_vec(label)))
    print("read %d sentences from %s ." % (len(data), sens_file_name))
    sens_file.close()
    label_file.close()
    return data

def read_dataset(sens_file_name, word_to_id):
    sens_file = open(sens_file_name)
    data = []
    for sens in sens_file:
        word_id_seq = map_token_seq_to_word_id_seq(tokenize(sens), word_to_id)
        data.append(word_id_seq)
    print("read %d sentences from %s ." % (len(data), sens_file_name))
    sens_file.close()
    return data

def main():
    sens_file_name = 'sentences_train.txt'
    label_file_name = 'labels_train.txt'
    correct_label = tf.placeholder(tf.float32, shape=[num_classes])
    word_to_id = build_vocab(sens_file_name)
    print(read_labeled_dataset(sens_file_name, label_file_name, word_to_id))
    #print(word_to_id)


if __name__ == "__main__":
   main()
