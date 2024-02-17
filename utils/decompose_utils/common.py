from utils.model_utils.modular_layers import Layer, LayerList





def recurse_layers(module, func):  # dfs
    func(module)
    if not isinstance(module, Layer):
        for i, layer in enumerate(module):
            recurse_layers(layer, func)



