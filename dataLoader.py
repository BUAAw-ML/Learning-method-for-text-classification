# -*- coding: utf-8 -*-

from random import shuffle
import numpy as np
import shutil
import os

import csv


def CrossValidationSplitter(file_path):

    data = []
    with open(file_path, newline='', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            content = line.strip().split('#')
            if len(content) != 4:
                continue

            data.append((content[0], content[1], content[2]))

    data_len = len(data)  # 获取文件总数
    shuffle(files)
    data_storage = []  # 初始化一个列表，用来接收分划分好的文件路径
    remainder = data_len % 10  # 判断文件数量能否直接被10整除
    if remainder == 0:  # 如果可以整除，直接将数据切成10组
        np_files = np.array(files)  # 将文件路径列表转换成numpy
        data_storage = np_files.reshape(10, -1)  # 利用numpy的reshape来将文件路径切分为10组
        # 比如说现在有20个文件路径
        # reshape()后得到的结果为2、2、2、2、2、2、2、2、2、2，即共十份、每份包含2个文件路径。
        return data_storage
    else:  # 否则，则先切开余数部分的文件
        np_files = np.array(files[:-1*remainder])  # 切开余数部分的文件，使文件数量保证能够被10整除
        data_storage_ten = np_files.reshape(10, -1)  # 同样利用上面的方法使用numpy切分10组文件
        # 获取余数部分的文件列表，遍历列表，尽可能的将多余的文件分散在10组文件中，而不是直接加入到一个文件中
        remainder_files = (
            np.array(files[-1*remainder:])).reshape(remainder, -1)  # 使用reshape切分问一份一组
        for i in range(0, len(remainder_files)):
            ech_dst = data_storage_ten[i]
            ech_rf = remainder_files[i]
            # 将取出来的余数内的路径分别加入到已经均分好的10份的前remainder个数据当中，比如说现在有24份文件，
            # 将24拆份拆分成一个能被10整除的数和一个余数，即这里拆分成20和4，我们首先将拆出来的20份文件均分10份，
            # 即每份有2个文件路径，然后，再将剩下后面的4个文件路径，尽可能的放入到刚刚均分好的10份数据中。
            # 因此最终拆分的结果共有十份，每份数量分别为：3、3、3、3、2、2、2、2、2、2。
            data_storage.append(np.concatenate((ech_dst, ech_rf)))
        for j in range(remainder, len(data_storage_ten)):
            # 将将剩下的没有被余数部分加入的份加入到data_storage中
            data_storage.append(data_storage_ten[j])
        return np.array(data_storage)


