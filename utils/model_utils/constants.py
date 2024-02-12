import enum


class ArchitectureType(enum.Enum):
    Transformer = 0
    Bert = 1
    GPT = 2


class LayerType(enum.Enum):
    Activation = 0
    Linear = 1
    Dropout = 2
    Embedding = 3
    ModuleList = 4
    MultiheadAttention = 5

    TransformerEncoder = 4
    TransformerDecoder = 5
    # NotRecognize = 7


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
    model_name = type(model).__name__
    if "Bert" in model_name:
        return ArchitectureType.Bert
    elif "Transformer" in model_name:
        return ArchitectureType.Transformer
    else:
        return ArchitectureType.GPT

def get_layer_type(layer):
    layer_name = type(layer).__name__
    return layer_name


def get_activation_type(layer):
    pass

