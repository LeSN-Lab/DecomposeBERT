import enum


class LayerType(enum.Enum):
    Dense = 1
    Embedding = 2
    LayerNorm = 3
    PositionalEncoding = 4
    MultiHeadAttention = 5
    PositionwiseFeedforward = 6
    NotRecognize = 7


class ActivationType(enum.Enum):
    Linear = 0
    Tanh = 1
    GeLU = 2
    Softmax = 3


def getLayerType(layer):
    layer_name = type(layer).__name__
    return layer_name


def getActivationType(layer):
    pass

