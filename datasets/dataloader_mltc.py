import csv
import copy
import os
import sys
from random import shuffle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
import json

tokenizer = BertTokenizer.from_pretrained("../datasets/bert-base-uncased")

def load_data(data_config, use_previousData=False):
    cache_file_head = data_config['dataset_name'] 
    if use_previousData:

        print("load dataset from cache")
        dataset = dataEngine.from_dict(torch.load(os.path.join('cache', 'datasets',  cache_file_head, cache_file_head + '.dataset')))

    else:
        print("build dataset")
        if not os.path.exists('cache'):
            os.makedirs('cache')

        dataset = dataEngine(data_config=data_config)

        file = os.path.join(data_config['dataset_path'], 'train.pkl')
        dataset.train_data = dataset.load_pkl(file)
        
        file = os.path.join(data_config['dataset_path'], 'test.pkl')
        dataset.test_data = dataset.load_pkl(file)
        
        os.makedirs(os.path.join('cache', 'datasets',  cache_file_head), exist_ok=True)
        torch.save(dataset.to_dict(), os.path.join('cache', 'datasets',  cache_file_head, cache_file_head + '.dataset'))

    return dataset


class dataEngine(Dataset):
    def __init__(self, train_data=None, unlabeled_train_data=None, test_data=None,
                    tag2id={}, id2tag={}, data_config={}):
        self.train_data = train_data
        self.unlabeled_train_data = unlabeled_train_data
        self.test_data = test_data

        self.tag2id = tag2id
        self.id2tag = id2tag

        self.data_config = data_config

    @classmethod
    def from_dict(cls, data_dict):
        return dataEngine(data_dict.get('train_data'),
                       data_dict.get('unlabeled_train_data'),
                       data_dict.get('test_data'),
                       data_dict.get('tag2id'),
                       data_dict.get('id2tag'))

    def to_dict(self):
        data_dict = {
            'train_data': self.train_data,
            'unlabeled_train_data': self.unlabeled_train_data,
            'test_data': self.test_data,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag
        }
        return data_dict

    def get_tags_num(self):
        return len(self.tag2id)

    def collate_fn(self, batch):

        # construct input
        inputs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e['text0'].strip())) for e in batch]
        
        # dscp_tokens = tokenizer.tokenize(dscp)

        # dscp_ids = tokenizer.convert_tokens_to_ids(dscp_tokens)

        #  = [e['tag_ids'] for e in batch]

        lengths = np.array([len(e) for e in inputs])
        max_len = np.max(lengths)  #_to_max_length=True , truncation=True
        if max_len > 510:
            max_len = 510
        # inputs = [tokenizer.prepare_for_model(e, max_length=max_len+2, pad_to_max_length=True) for e in inputs]
        inputs = [tokenizer.prepare_for_model(e, max_length=max_len + 2, pad_to_max_length=True, truncation=True) for e in inputs]

        ids = torch.LongTensor([e['input_ids'] for e in inputs])
        token_type_ids = torch.LongTensor([e['token_type_ids'] for e in inputs])
        attention_mask = torch.FloatTensor([e['attention_mask'] for e in inputs])
        
        # construct tag
        tags = torch.zeros(size=(len(batch), self.get_tags_num()))
        for i in range(len(batch)):
            tags[i, batch[i]['label']] = 1.

        return (ids, token_type_ids, attention_mask), tags

    def load_pkl(self, file):
        # TrainTest_programWeb_freecode_AAPD
        dataset = {}
        dataset["label_mapping"] = []
        data = []
        with open(file, 'rb') as pklfile:

            reader = pickle.load(pklfile)

            for row in reader:
                
                if len(row) != 4:
                    continue
                
                id = row["id"]
                dscp = row["name"] + row["descr"]
                tag = row["tags"]

                for t in tag:
                    if t not in self.tag2id:
                        tag_id = len(self.tag2id)
                        self.tag2id[t] = tag_id
                        self.id2tag[tag_id] = t
                        dataset["label_mapping"].append(t)

                tag_ids = [self.tag2id[t] for t in tag]

                data.append({
                    'text0': dscp,
                    'label': tag_ids,
                })

        dataset["cased"] = False 
        dataset["paraphrase_field"] = "text0"
        dataset["data"] = data 
        print(len(data))

        return dataset

