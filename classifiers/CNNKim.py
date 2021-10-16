import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import neural
from neural.util import Initializer
from neural.util import Loader

# CNN-Kim
class CNN(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, output_size,
                 dropout_p=0.5, pretrained=None, double_embedding=False, cuda_device=0):
        super(CNN, self).__init__()
        self.cuda_device = cuda_device
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels

        self.initializer = Initializer()
        # self.loader = Loader()

        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        # CNN
        self.conv13 = nn.Conv2d(1, word_out_channels, (3, word_embedding_dim))
        self.conv14 = nn.Conv2d(1, word_out_channels, (4, word_embedding_dim))
        self.conv15 = nn.Conv2d(1, word_out_channels, (5, word_embedding_dim))

        self.dropout = nn.Dropout(p=dropout_p)

        hidden_size = word_out_channels*3
        self.linear = nn.Linear(hidden_size, output_size)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)1p[
            ';lkjhgz '
        ]
        return x

    def forward(self, x, usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x,self.conv13)
        x2 = self.conv_and_pool(x,self.conv14)
        x3 = self.conv_and_pool(x,self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        output = self.linear(x)
        return output
        # 得到的是概率值，如果要预测值，需要sigmoid后，经过阈值筛选得到 (0,1)值

    def predict(self, x, usecuda=True):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv13)
        x2 = self.conv_and_pool(x, self.conv14)
        x3 = self.conv_and_pool(x, self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        output = self.linear(x)
        output = torch.sigmoid(output) > 0.5
        return output

    def features(self,x,usecuda=True,with_forward=False):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv13)
        x2 = self.conv_and_pool(x, self.conv14)
        x3 = self.conv_and_pool(x, self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        return (x,self.linear(self.dropout(x))) if with_forward else (x,None)

    def features_with_pred(self,x,usecuda=True):
        x = self.features(x,usecuda)
        output = self.linear(x)
        return torch.cat((x,output), 1)