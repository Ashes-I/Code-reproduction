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

    def forward(self, q, k, v, pad_mask=None):
        batch, seq_size, dimension = q.shape
        n_d = self.d_model // self.n_head
        q,k,v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, -1, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, -1, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.n_head, n_d).permute(0, 2, 1, 3)
        score = q@k.transpose(2,3)/math.sqrt(n_d)

        if pad_mask is not None:
            score = score.masked_fill(pad_mask == 0, -1e9)

        score = self.softmax(score)@v
        score = score.permute(0,2,1,3).contiguous().view(batch, -1, dimension)
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
        out = self.gamma * out + self.beta
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        _x = x
        x = self.attention(x, x, x, pad_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        return x

class Encoder(nn.Module):
    def __init__(self, vocal_size, max_len, d_model, hidden, n_head, n_encodelayer, device, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocal_size, d_model, max_len, dropout, device)
        self.Encoder_layers = nn.ModuleList([
                EncoderLayer(d_model, hidden, n_head, dropout)
                for _ in range(n_encodelayer)
        ])

    def forward(self, x, pad_mask):
        x = self.embedding(x)
        for layer in self.Encoder_layers:
            x = layer(x, pad_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, hidden, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, dec, enc, pad_mask, cross_pad_mask):
        """
        dec: the output of embeeding in decoder part
        enc: the output of encoder
        """
        _x = dec
        x = self.attention1(dec, dec, dec, pad_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        _x = x
        x = self.cross_attention(x, enc, enc, cross_pad_mask)
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocal_size, max_len, d_model, hidden, n_head, n_decodelayer, device, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocal_size, d_model, max_len, dropout, device)
        self.Decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, hidden, n_head, dropout)
            for _ in range(n_decodelayer)
        ])
        self.fc = nn.Linear(d_model, vocal_size)
    def forward(self, dec, enc, pad_mask, cross_pad_mask):
        dec = self.embedding(dec)
        for layer in self.Decoder_layers:
            dec = layer(dec, enc, pad_mask, cross_pad_mask)
        dec = self.fc(dec)
        return dec

class Transfomer_001(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_vocal_size, dec_cocal_size, max_len,
                        d_model, n_heads, hidden, n_layer, dropout, device):
        super(Transfomer_001, self).__init__()

        self.encoder = Encoder(enc_vocal_size, max_len, d_model, hidden, n_heads, n_layer, device, dropout)
        self.decoder = Decoder(dec_cocal_size, max_len, d_model, hidden, n_heads, n_layer, device, dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k
        return mask

    def make_seq_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k), diagonal=0).bool().to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        cross_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) \
                   * self.make_seq_mask(trg, trg)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, cross_mask)
        return out
