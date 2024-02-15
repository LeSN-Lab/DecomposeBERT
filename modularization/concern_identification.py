import torch.nn as nn


class ConcernIdentificationOutput:

    def feedforward(self, module, previous_state=None):
        if previous_state is None:
            previous_state = module.

    def attention(self, module, previous_state=None):
        pass
    def propagate_through_layer(self, layer, x_t, activation):
        if layer.layer_type == LayerType.Linear:
            layer.weight = self.propagate_through_dense(layer, x_t, activation)
        elif layer.layer_type == LayerType.Embedding:
            layer.weight = self.embedding_lookup(self, layer, x_t)
        elif layer.layer_type == LayerType.LayerNorm:
            pass
        elif layer.layer_type == LayerType.Dropout:
            pass



class ConcernIdentificationEncoder:
    def propagateThroughLayer(self, layer, x_t):
        if layer.act_fcn is ActivationType.Linear:
            if layer.layer_type is LayerType.Linear:
                layer.hidden_state = self.propagate_through_dense(layer, x_t=x_t,
                                                                apply_activation=is_active)
            elif layer.layer_type is LayerType.Embedding:
                return self.embedding_lookup(layer, x_t=x_t)
            elif layer.layer_type is LayerType.Dropout:
                return x_t

        return layer.hidden_state

    def propagate_through_activation(self, layer, x_t, is_active=True):
        if not is_active or layer.activation == ActivationType.Linear:
            return x_t
        if ActivationType.Softmax == layer.activation:
            x_t = x_t.reshape(layer.num_node)
            x_t = nn.Softmax(x_t)
        elif ActivationType.GELU == layer.activation:
            x_t = nn.GELU(x_t)
        elif ActivationType.Tanh == layer.activation:
            x_t = nn.Tanh(x_t)
        return x_t

    def embedding_lookup(self, layer, x):
        pass

class ConcernIdentificationDecoder:
    pass

class ConcernIdentificationEmbedding:
    pass