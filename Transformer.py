import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TokenEmbedding(nn.Embedding):
    '''
    convert the input vocabulary into embedding of the specified dimension
    vocab_size: the size of vocabulary
    d_model: the dimension of the embedding
    '''
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

