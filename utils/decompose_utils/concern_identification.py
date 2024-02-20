import copy

import torch


class ConcernIdentificationBert:
    def __init__(self):
        self.positive_sample = True

    def propagate(self, module, input_tensor, positive_sample=True):
        # propagate input tensor to the module
        self.positive_sample = positive_sample
        output_tensor = module.embeddings(input_tensor)
        output_tensor = self.propagate_encoder(module.encoder, output_tensor)
        output_tensor = self.propagate_pooler(module.pooler, output_tensor)
        output_tensor = module.dropout(output_tensor)
        output_tensor = self.propagate_classifier(module.classifier, output_tensor)
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
        # handle = module.register_forward_hook(encoder_hook)
        # for layer in module.encoder.layers:
        #     layer.register_forward_hook()
        output_tensor = module(input_tensor)
        # handle.remove()
        return output_tensor

    def propagate_pooler(self, module, input_tensor):
        def pooler_hook(module, input, output):
            if output.shape[0] > 1:
                raise "batch size error"

            target_layer = module.dense
            out_features, input_features = target_layer.shape
            original_weight, original_bias = target_layer.get_parameters()
            output_tensor = output.clone()
            output_tensor[output >= 0] = output[output >= 0]  # relu
            if not self.positive_sample:
                original_weight[original_weight > 0] = original_weight[
                    original_weight > 0
                ]

            non_zeros = 0

            for i in range(out_features):
                if output_tensor[0][i] > 0:
                    if self.positive_sample:  # if positive samples
                        target_layer.weight[i, :] = original_weight[i, :]
                    else:  # if negative samples
                        for j in range(input_features):
                            target_layer.weight[i, j] = (
                                torch.min(
                                    target_layer.weight[i, j],
                                    original_weight[i, j],
                                )
                                if original_weight[i, j] > 0
                                else torch.max(
                                    target_layer.weight[i, j],
                                    original_weight[i, j],
                                )
                            )
                    target_layer.bias[i] = original_bias[i]
                else:
                    target_layer.weight[i, :] = 0
                    target_layer.bias[i] = 0
            module.dense.set_parameters()

        handle = module.register_forward_hook(pooler_hook)
        output_tensor = module(input_tensor)
        handle.remove()
        return output_tensor

    @staticmethod
    def propagate_classifier(module, input_tensor):
        def classifier_hook(module, input, output):
            # for node_num in range(module.shape):
            pass

        handle = module.register_forward_hook(classifier_hook)
        output_tensor = module(input_tensor)
        handle.remove()
        return output_tensor
