from torch import nn
import collections
import math
import torch
from d2l import torch as d2l

class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_size)       # 行数=词元种类（词表大小），列数=特征向量维度
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        
    def forward(self, X, *args):
        X = self.embedding(X)
        print(X.shape)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state
    
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)        
        # state[-1]取最后一层的隐状态，形状变成[batch_size, num_hiddens];由于state[-1]只有一个时间步，需要用repeat(dim0, dim1, dim2)分别在三个维度上分别复制dim0, dim1, dim2次。此处dim0 = num_step(时间步数量)，dim1=dim2=1，表示保持原样，相当于把(batch_size, num_hiddens)复制了num_step份。
        X_cat_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_cat_context, state)
        output = self.dense(output).permute(1, 0, 2)

        return output, state
    
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = d2l.sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        
        return weighted_loss
        
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)       # 此处X不是元组，是Tensor

decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
# output：一个张量，表示每个时间步对词表的未归一化得分（logits）。形状为 (batch_size, num_steps, vocab_size)
# state：GRU 的隐藏状态张量，包含所有层的最终隐状态，形状为 (num_layers, batch_size, num_hiddens）