import csv
import copy
import os

from random import shuffle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

token_table = {'ecommerce': 'electronic commerce'}


class ProgramWebDataset(Dataset):
    def __init__(self, data, co_occur_mat, tag2id, id2tag=None):
        self.data = data
        self.co_occur_mat = co_occur_mat
        self.tag2id = tag2id
        if id2tag is None:
            id2tag = {v: k for k, v in tag2id.items()}
        self.id2tag = id2tag

    @classmethod
    def from_dict(cls, data_dict):
        return ProgramWebDataset(data_dict.get('data'),
            data_dict.get('co_occur_mat'),
            data_dict.get('tag2id'),
            data_dict.get('id2tag'))

    @classmethod
    def from_csv(cls, api_csvfile, net_csvfile):
        data, tag2id, id2tag = ProgramWebDataset.load(api_csvfile)
        co_occur_mat = ProgramWebDataset.stat_cooccurence(data, len(tag2id))
        #co_occur_mat = ProgramWebDataset.similar_net(net_csvfile, tag2id)
        return ProgramWebDataset(data, co_occur_mat, tag2id, id2tag)

    @classmethod
    def load(cls, f):
        data = []
        tag2id = {}
        id2tag = {}
        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                if len(row) != 4:
                    continue
                id, title, dscp, tag = row
                title_tokens = tokenizer.tokenize(title.strip())
                dscp_tokens = tokenizer.tokenize(dscp.strip())
                title_ids = tokenizer.convert_tokens_to_ids(title_tokens)
                dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)
                tag = tag.strip().split('###')
                tag = [t for t in tag if t != '']
                if len(tag) == 0:
                    continue
                for t in tag:
                    if t not in tag2id:
                        # tag_tokens = tokenizer.tokenize(t)
                        # if np.any([token.startswith('##') for token in tag_tokens]):
                        #     print(t, ':', tag_tokens)
                        tag_id = len(tag2id)
                        tag2id[t] = tag_id
                        id2tag[tag_id] = t
                tag_ids = [tag2id[t] for t in tag]
                data.append({
                    'id': int(id),
                    'title_ids': title_ids,
                    'title_tokens': title_tokens,
                    'dscp_ids': dscp_ids,
                    'dscp_tokens': dscp_tokens,
                    'tag_ids': tag_ids,
                })
        os.makedirs('cache', exist_ok=True)
        return data, tag2id, id2tag

    @classmethod
    def stat_cooccurence(cls, data, tags_num):
        co_occur_mat = torch.zeros(size=(tags_num, tags_num))
        for i in range(len(data)):
            tag_ids = data[i]['tag_ids']
            for t1 in range(len(tag_ids)):
                for t2 in range(len(tag_ids)):
                    co_occur_mat[tag_ids[t1], tag_ids[t2]] += 1
        return co_occur_mat

    @classmethod
    def similar_net(cls, csvfile, tag2id):

        tags_num = len(tag2id)
        co_occur_mat = torch.zeros(size=(tags_num, tags_num))

        with open(csvfile, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                if len(row) != 3:
                    continue
                tag1, similar, tag2 = row

                tag1 = tag1.strip()
                tag2 = tag2.strip()
                co_occur_mat[tag2id[tag1], tag2id[tag2]] += float(similar)

        return co_occur_mat

    def to_dict(self):
        data_dict = {
            'data': self.data,
            'co_occur_mat': self.co_occur_mat,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag
        }
        return data_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_tags_num(self):
        return len(self.tag2id)

    def encode_tag(self):
        tag_ids = []
        tag_token_num = []
        for i in range(self.get_tags_num()):
            tag = self.id2tag[i]
            tokens = tokenizer.tokenize(tag)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            tag_ids.append(token_ids)
            tag_token_num.append(len(tokens))
        max_num = max(tag_token_num)
        padded_tag_ids = torch.zeros((self.get_tags_num(), max_num), dtype=torch.long)
        mask = torch.zeros((self.get_tags_num(), max_num))
        for i in range(self.get_tags_num()):
            mask[i, :len(tag_ids[i])] = 1.
            padded_tag_ids[i, :len(tag_ids[i])] = torch.tensor(tag_ids[i])
        return padded_tag_ids, mask
        
    def collate_fn(self, batch):
        result = {}
        # construct input
        inputs = [e['title_ids'] + e['dscp_ids'] for e in batch]
        lengths = np.array([len(e) for e in inputs])
        max_len = np.max(lengths)
        inputs = [tokenizer.prepare_for_model(e, max_length=max_len+2, pad_to_max_length=True) for e in inputs]
        ids = torch.LongTensor([e['input_ids'] for e in inputs])
        token_type_ids = torch.LongTensor([e['token_type_ids'] for e in inputs])
        attention_mask = torch.FloatTensor([e['attention_mask'] for e in inputs])
        # construct tag
        tags = torch.zeros(size=(len(batch), self.get_tags_num()))
        for i in range(len(batch)):
            tags[i, batch[i]['tag_ids']] = 1.
        return (ids, token_type_ids, attention_mask), tags


# def CrossValidationSplitter(dataset, seed):
#     data_len = len(dataset)  # 获取文件总数

#     npr = np.random.RandomState(seed)

#     data_index = npr.permutation(data_len)

#     remainder = data_len % 10
#     data_index = data_index[:-1 * remainder]
#     data_index = np.array(data_index)
#     data_block = data_index.reshape(10, -1)  # split the data into 10 groups
#     return data_block


def build_dataset(api_csvfile=None, net_csvfile=None):
    if os.path.isfile('cache/ProgramWeb.state') and False:
        return ProgramWebDataset.from_dict(
            torch.load('cache/ProgramWeb.state'))
    else:
        dataset = ProgramWebDataset.from_csv(api_csvfile, net_csvfile)
        torch.save(dataset.to_dict(), 'cache/ProgramWeb.state')
        return dataset


def load_train_val_dataset(dataset):
    data = dataset.data
    train_dataset = dataset
    val_dataset = copy.copy(dataset)

    train_dataset.data = data[:-1000]
    val_dataset.data = data[-1000:]
    return train_dataset, val_dataset

