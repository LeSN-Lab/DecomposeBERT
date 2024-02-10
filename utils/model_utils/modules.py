import torch
import torch.nn as nn
import math
from utils.model_utils.layers import MultiHeadAttention, PositionwiseFeedforward, PositionalEncoding
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.multi_head_attention(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.layer_norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.layer_norm2(src)
        return src
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_attn_output = self.masked_multi_head_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt_attn_output)
        tgt = self.layer_norm1(tgt)

        enc_dec_attn_output = self.multi_head_attention(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(enc_dec_attn_output)
        tgt = self.layer_norm2(tgt)

        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.layer_norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, num_tokens, d_model, N, num_heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embed = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.tgt_embed = nn.Embedding(num_tokens, d_model)
        self.final_linear = nn.Linear(d_model, num_tokens)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        tgt = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)
        
        output = self.final_linear(tgt)
        return output
    

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, segment_types):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_types, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, num_attention_heads)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_after_attention = nn.LayerNorm(hidden_size)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_after_output = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, hidden_states, hidden_states, attention_mask)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.layer_norm_after_attention(hidden_states + attention_output)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output.dropout(layer_output)
        layer_output = self.layer_norm_after_output(attention_output + layer_output)
        return layer_output
    
class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings, segment_types, dropout_rate):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, segment_types)
        self.encoder_layers = nn.ModuleList([TransformerEncoder(hidden_size, num_attention_heads, intermediate_size, dropout_rate) for _ in range(num_hidden_layers)])
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = embedding_output
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, extended_attention_mask)

        pooled_output = self.pooler(encoder_output[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        return encoder_output, pooled_output

class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(GPTDecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply masked attention
        attn_output = self.masked_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Apply feedforward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x
    
class GPT(nn.Module):
    def __init__(self, num_tokens, d_model, N, num_heads, d_ff, dropout=0.1):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)
        ])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.final_layer = nn.Linear(d_model, num_tokens)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        logits = self.final_layer(x)
        return logits