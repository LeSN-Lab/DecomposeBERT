
import torch
import torch.nn as nn
from utils.model_utils.modules import TransformerBlock, BERTEmbedding
from utils.model_utils.modules import PositionwiseFeedForward

"""
Embedding layers
Attention Mask
Encoder layer
Multi-head attention
Scaled dot product attention
Position-wise feed-forward network
BERT (assembling all the components)
"""

class BERT(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size=512,
                 num_layers=6,
                 heads=8,
                 dropout=0.1,
                 forward_expansion=4,
                 device="cuda",
                 max_length=100):
        """
        :param vocab_size: vocabulary size of total words
        :param embed_size: BERT model embedding(hidden) size
        :param num_layers: the number of Transformer blocks(layers)
        :param heads: the number of attention heads
        :param dropout:
        :param device:
        :param forward_expansion:
        :param max_length:
        """

        super(BERT, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.ff_hidden = embed_size * 4
        self.word_embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=embed_size)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size=embed_size, attn_heads=heads, d_ff = self.ff_hidden, dropout=dropout) for _ in range(num_layers)])

    def forward(self, input_ids, segment_info):
        """
        :param input_ids: Input token ids of shape (batch_size, sequence_length)
        :param attn_mask: Mask to avoic performing attention on podding tokens.
        :return: The las hidden state of the BERT model, with shape (batch_size, sequence_length, embed_size)
        """
        mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)

        # Get word embeddings for input tokens
        input_ids = self.word_embedding(input_ids, segment_info)

        # Add positional encoding
        for transformer_block in self.transformer_blocks:
            input_ids = transformer_block(input_ids, mask)

        return input_ids
