import torch.nn as nn
from utils.model_utils.layers import LayerType
from transformers import BertModel
from transformers.activations import get_activation


def sigmoid(x):
    return get_activation('sigmoid')(x)


def tanh(x):
    return get_activation('tanh')(x)


def relu(x):
    return get_activation('relu')(x)

def linear(x):
    return get_activation('linear')(x)

def gelu(x):
    return get_activation('gelu')(x)

def initModularLayers(layers):
    pass

def getLayerType(layer):
    if isinstance(layer, nn.Embedding):
        return LayerType.Embedding
    elif isinstance(layer, nn.Linear):
        return LayerType.Linear
    elif isinstance(layer, nn.LayerNorm):
        return LayerType.LayerNorm
    elif isinstance(layer, nn.Dropout):
        return LayerType.Dropout
    elif isinstance(layer, BertModel.BertSelfAttention):
        return LayerType.BertSelfAttention
    elif isinstance(layer, BertModel.BertLayer):
        return LayerType.BertLayer
    elif isinstance(layer, BertModel.BertBertSelfOutput):
        return LayerType.BertSelfOutput
    elif isinstance(layer, BertModel.BertIntermediate):
        return LayerType.BertIntermediate
    elif isinstance(layer, BertModel.BertOutput):
        return LayerType.BertOutput
    elif isinstance(layer, BertModel.BertPooler):
        return LayerType.BertPooler
    else:
        return LayerType.Other