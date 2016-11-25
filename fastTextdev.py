#!/usr/bin/python

import collections
import math
import os
import random

import numpy as np
import tensorflow as tf
from random import shuffle

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

##
from nltk import word_tokenize
from nltk.corpus import stopwords

# nltk predefined english stop words
stop_words = set(stopwords.words('english'))


def create_label_vec(label):
   # Generate a label vector for a given classification label.
   # return label_to_id[label.rstrip()]
   temp = [0,0,0]
   temp[label_to_id[label.rstrip()]] = 1
   # temp[label_to_id[label[0]]] = 1
   return temp



def tokenize(sens):
    # Tokenize a given sentence into a sequence of tokens.
    # word_tokens = word_tokenize(sens)
    # filtered_sens = [w for w in word_tokens if not w in stop_words]
    # return filtered_sens
    return word_tokenize(sens)

def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id,word) for word in token_seq]


def map_word_to_id(word_to_id, word):
    # map each word to its id
    '''if the word is in the dictionary
            then return the index
                else return 0'''
    if word in word_to_id:
        return word_to_id[word]
    else:
        return unknown_word_id
        # return word_to_id['$UNK$']


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
        # data.append((word_id_seq, create_label_vec(tokenize(label))))
        data.append((word_id_seq, create_label_vec(label)))
        # print(create_label_vec())
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


def eval(word_to_id, train_dataset, dev_dataset, test_dataset):
    # Initialize the placeholders and Variables. E.g.
    num_words = len(word_to_id)
    # correct_label = tf.placeholder(tf.float32, shape=[num_classes])
    input_sens = tf.placeholder(tf.int32, shape = [None])
    correct_label = tf.placeholder(tf.float32, shape= [num_classes])
    # Word embeddings are the only parameters of the model
    embeddings = tf.Variable(tf.random_uniform([num_words, embedding_dim], -1.0, 1.0))  # [100, 10]
    # Hint: use [None] when you are not certain about the value of shape
    test_results = []

    with tf.Session() as sess:
        embed = tf.nn.embedding_lookup(embeddings, input_sens)
        print('embed',embed)
        tmp_m = tf.reduce_sum(embed, 0)
        print('tmp_m',tmp_m)
        sum_rep = tf.reshape(tmp_m, [1, embedding_dim])
        print('sum_rep',sum_rep)
        # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.

        W = tf.Variable(tf.zeros([embedding_dim, num_classes]))
        b = tf.Variable(tf.zeros([num_classes]))
        y = tf.nn.softmax(tf.matmul(sum_rep, W) + b) # [1,3]

        # y = tf.nn.softmax(tf.matmul(sum_rep, embeddings, transpose_b = True))
        # y = tf.nn.softmax(tf.matmul(sum_rep, embeddings, transpose_b = True))
        print('log(y)', tf.log(y))   # (1, 3)
        print('correct_label', correct_label)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y)*correct_label, reduction_indices=[1]))
        print('cross_entropy ', cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(correct_label, 0))
        print('here')
        accuracy = tf.cast(correct_prediction, tf.float32)
        print('1here')
        prediction = tf.cast(tf.argmax(y, 1), tf.int32)
        print('2here')


        sess.run(tf.initialize_all_variables())

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        for epoch in range(num_epochs):
            shuffle(train_dataset)
            for (sens, label) in train_dataset:
                train_step.run(feed_dict={input_sens: sens, correct_label: label})
            print('Epoch %d : %s .' % (epoch, compute_accuracy(accuracy, input_sens, correct_label, dev_dataset)))

        # uncomment the following line in the grading lab for evaluation
        print('Accuracy on the test set : %s.' % compute_accuracy(accuracy,input_sens, correct_label, test_dataset))
        # input_sens is the placeholder of an input sentence.
        test_results = predict(prediction, input_sens, test_dataset)
    return test_results


def compute_accuracy(accuracy, input_sens, correct_label, eval_dataset):
    num_correct = 0
    for (sens, label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_sens: sens, correct_label: label})
    print('#correct sentences is %s ' % num_correct)
    return num_correct / len(eval_dataset)


def predict(prediction, input_sens, test_dataset):
    test_results = []
    for (sens, label) in test_dataset:
        test_results.append(prediction.eval(feed_dict={input_sens: sens}))
    return test_results


def write_result_file(test_results, result_file):
    with open(result_file, mode='w') as f:
         for r in test_results:
             f.write("%d\n" % r)


def main(argv):
    trainSensFile = ''
    trainLabelFile = ''
    devSensFile = ''
    devLabelFile = ''
    testSensFile = ''
    testLabelFile = ''
    testResultFile = ''
    try:
        opts, args = getopt.getopt(argv,"hd:",["dataFolder="])
    except getopt.GetoptError:
        print('fastText.py -d <dataFolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fastText.py -d <dataFolder>')
            sys.exit()
        elif opt in ("-d", "--dataFolder"):
            trainSensFile = os.path.join(arg, 'sentences_train.txt')
            devSensFile = os.path.join(arg, 'sentences_dev.txt')
            testSensFile = os.path.join(arg, 'sentences_test.txt')
            trainLabelFile = os.path.join(arg, 'labels_train.txt')
            devLabelFile = os.path.join(arg, 'labels_dev.txt')
            # uncomment the following line in the grading lab
            testLabelFile = os.path.join(arg, 'labels_test.txt')
            testResultFile = os.path.join(arg, 'test_results.txt')
        else:
            print("unknown option %s ." % opt)
 ## Please write the main procedure here by calling appropriate methods.
    word_to_id = build_vocab(trainSensFile)
    train_dataset = read_labeled_dataset(trainSensFile, trainLabelFile, word_to_id)
    dev_dataset = read_labeled_dataset(devSensFile, devLabelFile, word_to_id)
    test_dataset = read_labeled_dataset(testSensFile, testLabelFile, word_to_id)
    test_results = eval(word_to_id, train_dataset, dev_dataset, test_dataset)

if __name__ == "__main__":
   main(sys.argv[1:])
