import math

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TokenEmbedding(nn.Embedding):
    """
    convert the input vocabulary into embedding of the specified dimension
    vocab_size: the size of vocabulary
    d_model: the dimension of the embedding
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model,padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000 ** (_2i/d_model)))

    def forward(self, x):
        batch_size, seq_size = x.size()
        return self.encoding[:seq_size, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocal_size, d_model, max_len, drop_out, device):
        super(TransformerEmbedding, self).__init__()
        self.token_Emb = TokenEmbedding(vocal_size, d_model)
        self.pos_Emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        token_Emb = self.token_Emb(x)
        pos_Emb = self.pos_Emb(x)
        return self.drop_out(token_Emb + pos_Emb)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, pad_mask=None, seq_mask=None):
        batch, seq_size, dimension = q.shape
        n_d = self.d_model // self.n_head
        q,k,v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, seq_size, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, seq_size, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, seq_size, self.n_head, n_d).permute(0, 2, 1, 3)
        score = q@k.transpose(2,3)/math.sqrt(n_d)

        if seq_mask is not None:
            score = score.masked_fill(seq_mask, -1e9)

        # 再应用 Padding Mask（遮挡填充符）
        if pad_mask is not None:
            score = score.masked_fill(pad_mask == 0, -1e9)

        score = self.softmax(score)@v
        score = score.permute(0,2,1,3).contiguous().view(batch, seq_size, dimension)
        out = self.w_combine(score)
        return out


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x-mean) / torch.sqrt(var+self.eps)
        out = out * self.gamma + self.beta
        return out

