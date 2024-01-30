import torch
from torch import nn
import torch.nn.functional as F

class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, segment_type_size=2):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_type_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

class LayerType(enum.Enum):
    Embedding = 1
    Linear = 2
    LayerNorm = 3
    Dropout = 4
    BertLayer = 5
    BertSelfAttention = 6
    BertSelfOutput = 7
    BertIntermediate = 8
    BertOutput = 9
    BertPooler = 10
    Other = 11

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1)

    def forward(self, hidden_states):
        """
        :param hidden_states: shape of hidden states is [batch_size, sequence_length, hidden_size]
        :return:
        """
        hidden_states = hidden_states.transpose(0, 1)
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        return attention_output.transpose(0, 1)



    def forward(self, input_feature):
        return self.mlp