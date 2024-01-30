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

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(token_type_ids)

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


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.scaled_dot_product_attention = ScaledDotProductAttention()

        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads x head_dim).
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        attention_output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embed_dim)

        output = self.out_linear(attention_output)

        return output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.attention_output_dropout = nn.Dropout(dropout_rate)
        self.attention_output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        # MultiheadAttention in PyTorch expects inputs in the shape of [sequence_length, batch_size, hidden_size]
        hidden_states = hidden_states.transpose(0, 1)

        # Attention mechanism
        attention_output, _ = self.multihead_attention(hidden_states, hidden_states, hidden_states, key_padding_mask=attention_mask)
        attention_output = self.attention_output_dropout(attention_output)
        attention_output = self.attention_output_layer_norm(attention_output + hidden_states)

        # Back to original shape for feed-forward network
        attention_output = attention_output.transpose(0, 1)

        # Feed-forward network
        intermediate_output = F.relu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        super().__init__()
        # The first linear layer expands the input to a larger dimension (intermediate_size)
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        # The second linear layer projects it back to the hidden_size
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply the first linear transformation
        x = self.linear_1(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Apply dropout for regularization
        x = self.dropout(x)
        # Apply the second linear transformation
        x = self.linear_2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x[:, 0])  # Assuming using the output corresponding to [CLS] token for classification


# Example usage
hidden_size = 768
num_attention_heads = 12
intermediate_size = 3072
encoder_layer = TransformerEncoderLayer(hidden_size, num_attention_heads, intermediate_size)

# Example input (batch_size, seq_length, hidden_size)
example_input = torch.rand(32, 128, hidden_size)
attention_mask = torch.ones(32, 128).bool()  # Replace with actual mask

# Forward pass
output = encoder_layer(example_input, attention_mask)

# Example usage
hidden_size = 768
intermediate_size = 3072  # This is typically larger than hidden_size
ffn = PositionwiseFeedforward(hidden_size, intermediate_size)

# Example input (batch_size, seq_length, hidden_size)
example_input = torch.rand(32, 128, hidden_size)

# Forward pass
output = ffn(example_input)