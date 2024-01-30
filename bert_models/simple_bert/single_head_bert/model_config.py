from utils.model_utils.layers import SingleHeadAttention
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

"""Embedding layers
Attention Mask
Encoder layer
Multi-head attention
Scaled dot product attention
Position-wise feed-forward network
BERT (assembling all the components)"""
class SingleHeadBertModel(nn.Module):
    def __init__(self, original_bert_model):
        super().__init__()
        self.bert = original_bert_model
        self.single_head_attention = SingleHeadAttention(self.bert.config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.single_head_attention(sequence_output)
        return sequence_output

print(torch.arange(30, dtype=torch.long).expand_as(input_ids))
original_bert_model = BertModel.from_pretrained('bert-base-uncased')
single_head_model = SingleHeadBertModel(original_bert_model)
