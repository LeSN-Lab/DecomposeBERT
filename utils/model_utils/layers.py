from utils.common import LayerType, getLayerType, getActivationType
import torch


def initEmbeddingLayers(embeddings):
    for name, layer in embeddings.named_children():
        if name == "word_embeddings":
            pass
        elif name == "position_embeddings":
            pass
        elif name == "token_type_embeddings":
            pass
        elif name == "LayerNorm":
            pass
        elif name == "dropout":
            pass


def initEncoderLayers(encoder):
    for layer_num, layer in enumerate(encoder.layer):
        attention = layer.attention
        print(attention.self)
        print(attention.output)


class ModularLayer:
    """
    Initialize each layer
    """

    type = LayerType.NotRecognize
    _W = None  # Weights
    _B = None  # Bias
    num_nodes = None  # The number of nodes
    activationType = None  # Activation Type of Layer
    is_first = False
    is_last = False

    def __init__(self, layer):
        self.name = layer.name
        self.type = getLayerType(layer)
        self.num_nodes = layer.output_shape[len()]
