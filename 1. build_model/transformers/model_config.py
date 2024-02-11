import torch.nn as nn
from utils.model_utils.layers import TransformerDecoder, TransformerEncoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_layers, n_heads, d_ff, dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output
