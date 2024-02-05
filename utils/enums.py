import enum

class LayerType(enum.Enum):
    # token embedding
    word_embeddings = 1
    # position embedding
    position_embeddings = 2
    # segment embedding
    token_type_embeddings = 3
    # LyayerNorm
    LayerNorm = 4

    Dropout = 5
    query = 6
    key = 7
    value = 8
    dense = 9
    activation = 10
    intermediate_act_fn = 11

class ActivationType(enum.Enum):
    Tanh = 1
    GeLU = 2

def getLayerType(layer):
    layer_name = type(layer).__name__
    return layer_name