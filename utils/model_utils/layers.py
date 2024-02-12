
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_utils.constants import LayerType, get_layer_type, get_activation_type


class ModularLayer(nn.Module):

    def __init__(self, layer):
        super(ModularLayer, self).__init__()
        self.type = get_layer_type(layer)
        self.num_node
        self.activation = get_activation_type(layer)
        self.setInputShape(layer)

        if self.type != LayerType.Activation:

            if self.type in []:
                pass
            elif self.type == LayerType.Dense:
                self.layer = nn.Linear()
            elif self.type == LayerType.Embedding:
                self.layer = nn.Embedding()
            elif self.type == LayerType.LayerNorm:
                self.layer = nn.LayerNorm()
                pass

    def forward(self, x):
        x = self.alyer(x)
        if self.activation:
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activation == 'gelu':
                x = torch.gelu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
        return x
        
    def get_weight(self):
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data
        return weights
    
    def set_weight(self, weights):
        for name, param in self.named_parameters():
            if name in weights:
                param.data = weights[name]

    def set_input_shape(self, layer):
        if self.type == LayerType.Embedding:
            pass
        elif self.type == LayerType.Linear:
            pass
        elif self.type == LayerType.LayerNorm:
            pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Define linear projection for Q, K, V
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Final linear layer to project output of concatenated heads
        self.final_projection = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.split_heads(self.query_projection(query), batch_size)
        key = self.split_heads(self.key_projection(key), batch_size)
        value = self.split_heads(self.value_projection(value), batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.Softmax(scores, dim=-1)
        attention = self.dropout(attention)
        # Apply the attention to the values and combine the heads
        output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.final_projection(output)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

    
