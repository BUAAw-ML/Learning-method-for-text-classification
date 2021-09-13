from torch.nn import Parameter
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch.nn.functional as F


class MLPBert(nn.Module):
    def __init__(self, output_dim, hidden_dim, hidden_layer_num, bert_trainable=True):
        super(MLPBert, self).__init__()
        
        bert = BertModel.from_pretrained('../datasets/bert-base-uncased') #cache/common/transformers_pretrained/bert-base-uncased
        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False
    
        self.act = nn.ReLU()
        self.Linear1 = nn.Linear(768, 512)
        self.Linear2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, ids, token_type_ids, attention_mask, return_feature=False, return_penultimateLayer=False):

        token_feat = self.bert(ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
            / torch.sum(attention_mask, dim=1, keepdim=True)
        
        if return_feature:
            return sentence_feat

        x = sentence_feat

        x = self.Linear1(x)
        e1 = self.act(x)
        x = self.dropout(e1)
        x = self.Linear2(x)

        y = torch.sigmoid(x)

        if return_penultimateLayer:
            #e1: [batchsize,512] y: [batchsize,label_num] -> [batchsize, 2]
            return torch.cat((y, e1),-1)

        return y
    
    def get_config_optim(self, lr, lrp=0.1):
        return [
                {'params': self.bert.parameters(), 'lr': lr * lrp}, #lr * lrp
                {'params': self.Linear1.parameters(), 'lr': lr},
                {'params': self.Linear2.parameters(), 'lr': lr},
                ]