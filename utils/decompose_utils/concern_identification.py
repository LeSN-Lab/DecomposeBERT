
class ConcernIdentificationBert:
    def __init__(self, config):
        self.config = config
        self.activations = []

    def propagate(self, module, input_tensor):
        # propagate input tensor to the module
        output_tensor = module.embeddings(input_tensor)
        output_tensor = module.encoder(output_tensor)
        output_tensor = module.pooler(output_tensor)
        output_tensor = module.dropout(output_tensor)
        output_tensor = module.classifier(output_tensor)
        return output_tensor

    @staticmethod
    def propagate_encoder(module, input_tensor):
        # check heads, concern and prune head
        # def encoder_hook(module, input, output):
        #     pass
        #
        # def attention_hook(module, input, output):
        #     pass
        #
        # def output_hook(module, input, output):
        #     pass
        #
        # module.register_forward_hook(encoder_hook)
        # for layer in module.encoder.layers:
        #     layer.register_forward_hook()
        output_tensor = module(input_tensor)
        # module.remove()
        return output_tensor

    @staticmethod
    def propagate_pooler(module, input_tensor):
        def pooler_hook(module, input, output):
            print(module.dense.shape)
            pass
        module.register_forward_hook(pooler_hook)
        output_tensor = module(input_tensor)
        module.remove()
        return output_tensor

    @staticmethod
    def propagate_classifier(module, input_tensor):
        def classifier_hook(module, input, output):
            pass
        module.register_forward_hook(classifier_hook)
        output_tensor = module(input_tensor)
        module.remove()
        return output_tensor

