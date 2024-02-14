import enum


class ArchitectureType(enum.Enum):
    Transformer = 0
    Bert = 1
    GPT = 2


class LayerType(enum.Enum):
    NotRecognize = 0
    Activation = 1
    Linear = 2
    Dropout = 3
    Embedding = 4
    LayerNorm = 5
    Attention = 6
    # MultiheadAttention = 6
    # TransformerEncoder = 6
    # TransformerDecoder = 7
    # GPT2Block = 8
    # GPT2Attention = 18


class ActivationType(enum.Enum):
    # Non-linear Activations
    Not = 0
    ReLU = 1
    Tanh = 2
    GELU = 3
    Softmax = 4


