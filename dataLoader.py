# -*- coding: utf-8 -*-

from random import shuffle
import numpy as np
import shutil
import os

import csv


def CrossValidationSplitter(file_path, seed):

    data = []
    with open(file_path, newline='', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        #no, APIName, descr, tags2
        for line in reader:
            if len(line) != 4:
                continue
            data.append((line[1], line[2], line[3]))

    data_len = len(data)  # 获取文件总数

    npr = np.random.RandomState(seed)

    data_index = npr.permutation(data_len)

    data_storage = []
    remainder = data_len % 10
    data_index = data_index[:-1*remainder]
    data_index = np.array(data_index)
    data_block = data_index.reshape(10, -1)  #split the data into 10 groups
    return data, data_block


def loadData(data, data_block, valData_pos):
    train_dataset = [data[no] for block in range(10) for no in data_block[block] if block != valData_pos]
    val_dataset = [data[no] for no in data_block[valData_pos]]
    return train_dataset, val_dataset



