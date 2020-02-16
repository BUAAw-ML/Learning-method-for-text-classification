import csv
import copy
import os

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from random import shuffle

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(batch):
    result = {}
    # construct input
    inputs = [e['title_ids'] + e['dscp_ids'] for e in batch]
    lengths = np.array([len(e) for e in inputs])
    max_len = np.max(lengths)
    inputs = [tokenizer.prepare_for_model(e, max_length=max_len, pad_to_max_length=True) for e in inputs]
    ids = torch.tensor([e['input_ids'] for e in inputs])
    token_type_ids = torch.tensor([e['token_type_ids'] for e in inputs])
    attention_mask = torch.tensor([e['attention_mask'] for e in inputs])
    # construct tag
    tags = torch.tensor([e['tag'] for e in batch])
    return (ids, token_type_ids, attention_mask), tags


class ProgramWebDataset(Dataset):
    def __init__(self, csvfile):
        self.id2tag = {}
        self.tag2id = {}
        self.data = self.load(csvfile)
        self.co_occur_mat = self.stat_cooccurence()

    def load(self, f):
        cache_file = 'cache/ProgramWeb.cache'
        if os.path.isfile(cache_file):
            print('Load cache data: %s' % cache_file)
            return torch.load(cache_file)
        data = []
        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) != 4:
                    continue
                id, title, dscp, tag = row
                title_tokens = tokenizer.tokenize(title)
                dscp_tokens = tokenizer.tokenize(dscp)
                title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)
                tag = tag.split('###')
                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t
                onehot_tag = torch.zeros(size=(len(self.tag2id),), dtype=torch.long)
                tag_ids = [self.tag2id[t] for t in tag]
                onehot_tag[tag_ids] = 1
                data.append({
                    'id': id,
                    'title_ids': title_ids,
                    'dscp_ids': dscp_ids,
                    'tag': onehot_tag
                })
        os.makedirs('cache', exist_ok=True)
        torch.save(data, cache_file)
        return data

    def stat_cooccurence(self):
        co_occur_mat = torch.zeros(size=(len(self.tag2id), len(self.tag2id)))
        for i in range(len(self.data)):
            tag = self.data[i]['tag']
            tag = torch.where(tag == 1)[0]
            for t1 in len(tag):
                for t2 in len(tag):
                    co_occur_mat[tag[t1], tag[t2]] += 1
        return co_occur_mat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def CrossValidationSplitter(dataset, seed):
    data_len = len(dataset.data)  # 获取文件总数

    npr = np.random.RandomState(seed)

    data_index = npr.permutation(data_len)

    remainder = data_len % 10
    data_index = data_index[:-1 * remainder]
    data_index = np.array(data_index)
    data_block = data_index.reshape(10, -1)  # split the data into 10 groups
    return data_block


def load_train_val_dataset(dataset, data_block, valData_block):
    data = dataset.data
    train_dataset = dataset
    val_dataset = copy.copy(dataset)

    train_dataset.data = [data[no] for block in range(10) for no in data_block[block] if block != valData_block]
    val_dataset.data = [data[no] for no in data_block[valData_block]]

    return train_dataset, val_dataset

