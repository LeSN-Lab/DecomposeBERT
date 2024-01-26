from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.attention import Attention, MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBert(nn.Module):5467321
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_hidden_layers, num_classes):
        super(SimpleBert, self).__init__()

        # create embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # bert encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                num_encoder_layers=num_hidden_layers,
            ),
            num_layers=num_hidden_layers,
        )

        #
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # 입력 데이터를 임베딩
        embedded = self.embedding(input_ids)

        # BERT 인코더에 입력
        encoded = self.encoder(embedded)

        # 마지막 토큰에 대한 표현을 추출
        pooled_output = encoded[:, 0, :]

        # 분류 레이어 적용
        logits = self.fc(pooled_output)
        return logits
