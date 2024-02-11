from utils.model_utils.constants import LayerType, ActivationType
import torch.nn as nn

class ConcernIdentification:
    def feedback_loop(self, layer):
        pass
    def propagateThroughLayer(self, layer, x_t, activation):
        if layer.type == LayerType.Dense:
            layer.hidden_state = self.propagateThroughDense(layer, x_t, activation)
        elif layer.type == LayerType.Embedding:
            layer.hidden_state = self.embeddingLookup(self, layer, x_t)
        elif layer.type == LayerType.MultiHeadAttention:
            pass
        elif layer.type == LayerType.LayerNorm:
            pass
        elif layer.type == LayerType.PositionalEncoding:
            pass
        elif layer.type == LayerType.PositionwiseFeedforward:
            pass
        
    def propagateThroughDense(self, layer, x_t, is_active=True):
        x_t = x_t.dot(layer.W) + layer.B
        return self.propagateThroughActivation(layer, x_t, is_active)
    
    def propagateThroughActivation(self, layer, x_t, is_active):
        if not is_active or layer.activation == ActivationType.Linear:
            return x_t
        if ActivationType.Softmax == layer.activation:
            x_t = x_t.reshape(layer.num_node)
            x_t = nn.Softmax(x_t)
        elif ActivationType.GeLU == layer.activation:
            x_t = nn.GELU(x_t)
        elif ActivationType.tanh == layer.activation:
            x_t = nn.Tanh(x_t)
        return x_t