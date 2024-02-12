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
    ModuleList = 5
    MultiheadAttention = 6
    TransformerEncoder = 7
    TransformerDecoder = 8
    GPT2Block = 9
    GPT2Attention = 10



class ActivationType(enum.Enum):
    # Linear Activations
    Linear = 0
    # Non-linear Activations
    ReLU = 1
    Tanh = 2
    GELU = 3
    Softmax = 4
    # Norm
    LayerNorm = 5

def get_architecture_type(model):
    if "Bert" in model:
        return ArchitectureType.Bert
    elif "Transformer" in model:
        return ArchitectureType.Transformer
    elif "GPT" in model:
        return ArchitectureType.GPT

def get_layer_type(layer):
    layer_name = type(layer).__name__
    return layer_name


def get_activation_type(layer):
    pass

