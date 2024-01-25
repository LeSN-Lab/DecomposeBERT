import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_position_embeddings, type_vocab_size):
        super(BERTEmbedding, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        