import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer ( self-attnetion)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, embed_size, attn_heads, d_ff, dropout):
        """
        :param embed_size: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param d_ff: dimensions of feed forward
        :param dropout: dropout rate
        """

        super(TransformerBlock).__init__()
        self.attention = nn.MultiheadAttention(num_heads=attn_heads, embed_dim=embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        # self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        # self.drop2 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=embed_size, d_ff=d_ff, dropout=dropout)
    def forward(self, x, mask):
        x = self.input_sublayer(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: dimensions of model
        :param d_ff: dimensions of feed forward
        :param dropout: a ratio of dropout
        """

        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(BERTEmbedding).__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super(TokenEmbedding).__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

