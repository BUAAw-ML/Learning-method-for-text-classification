from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNBert(nn.Module):
    def __init__(self, bert, num_classes, t=0, co_occur_mat=None, bert_trainable=True):
        super(GCNBert, self).__init__()
        
        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False
        
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(768, 768)
        self.gc2 = GraphConvolution(768, 768)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, co_occur_mat)
        _adj = torch.FloatTensor(_adj).transpose(0, 1)
        self.adj = nn.Parameter(gen_adj(_adj), requires_grad=False)

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask):
        token_feat = self.bert(ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
            / torch.sum(attention_mask, dim=1, keepdim=True)

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
            / torch.sum(tag_mask, dim=1, keepdim=True)
        x = self.gc1(tag_embedding, self.adj)
        x = self.relu(x)
        x = self.gc2(x, self.adj)

        x = x.transpose(0, 1)
        x = torch.matmul(sentence_feat, x)

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.bert.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


class MLPBert(nn.Module):
    def __init__(self, bert, num_classes, hidden_dim, hidden_layer_num, bert_trainable=True):
        super(MLPBert, self).__init__()
        
        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False
        
        self.num_classes = num_classes
        self.hidden_layer_num = hidden_layer_num
        self.hidden_list = nn.ModuleList()
        for i in range(hidden_layer_num):
            if i == 0:
                self.hidden_list.append(nn.Linear(768, hidden_dim))
            else:
                self.hidden_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask):
        token_feat = self.bert(ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
            / torch.sum(attention_mask, dim=1, keepdim=True)
        
        x = sentence_feat
        for i in range(self.hidden_layer_num):
            x = self.hidden_list[i](x)
            x = self.act(x)
        y = self.output(x)
        return y
    
    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.bert.parameters(), 'lr': lr * lrp},
                {'params': self.hidden_list.parameters(), 'lr': lr},
                {'params': self.output.parameters(), 'lr': lr},
                ]


class MABert(nn.Module):
    def __init__(self, bert, num_classes, hidden_dim, hidden_layer_num, bert_trainable=True):
        super(MABert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(tag_mask, dim=1, keepdim=True)

        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(1 - masks.byte(), torch.tensor(-np.inf))
        attention = F.softmax(attention, -1)
        attention_out = attention @ token_feat   # N, labels_num, hidden_size

        pred = torch.sum(attention_out, -1)

        return pred

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lr * lrp},
            {'params': self.hidden_list.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]


