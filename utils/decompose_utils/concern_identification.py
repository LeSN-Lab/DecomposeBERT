import torch
from utils.model_utils.modular_layers import get_extended_attention_mask
from utils.type_utils.data_type import safe_std
from scipy.stats import norm

class ConcernIdentificationBert:
    def __init__(self, model_config):
        self.positive_sample = True
        self.num_labels = model_config.num_labels

    def remove(self, module, output):
        """
        Positive hook
        Attributes:
            module (Layer): custom layer
            output (torch.Tensor): output tensor of the original layer
        """
        # Get the shapes and original parameters (weights and biases) of the layer
        current_weight, current_bias = module.weight, module.bias   # updating parameters
        upper_z = norm.ppf(0.75)
        mean_output = torch.mean(output, keepdim=True, dim=1)
        normalized_output = (output - mean_output) / safe_std(output, dim=1, epsilon=1e-5)
        mask = torch.abs(normalized_output) < upper_z
        temp = mask[0]
        for i in range(len(output[:])):
            temp = torch.logical_and(temp, mask[i])

        expanded_temp = temp.unsqueeze(1).expand(-1, module.shape[1])
        temp = expanded_temp
        # temp = torch.logical_and(expanded_temp, negative_extended_mask)
        current_weight[temp] = 0
        all_zeros = ~temp.any(dim=1)
        current_bias[all_zeros] = 0

        module.set_parameters(current_weight, current_bias)

    def pruning(self, module, original_output, output):
        # Get the shapes and original parameters (weights and biases) of the module

        current_weight, current_bias = module.weight, module.bias
        original_weight, original_bias = module.get_parameters()
        normalized_output = output.clone()

        positive_weight_mask = original_weight > 0
        negative_weight_mask = original_weight < 0
        upper_z = norm.ppf(0.60)

        for b in range(len(output[:])):
            single_output = output[b]
            output_loss = original_output[b] - single_output[b]
            positive_loss_mask = output_loss > 0
            negative_loss_mask = output_loss < 0
            positive_loss_mean = torch.mean(single_output[positive_loss_mask])
            negative_loss_mean = torch.mean(single_output[negative_loss_mask])

            positive_loss_std = safe_std(single_output[positive_loss_mask], dim=0, epsilon=1e-5)
            negative_loss_std = safe_std(single_output[negative_loss_mask], dim=0, epsilon=1e-5)

            normalized_output[b][positive_loss_mask] = (single_output[positive_loss_mask] - positive_loss_mean)/positive_loss_std
            normalized_output[b][negative_loss_mask] = (single_output[negative_loss_mask] - negative_loss_mean)/negative_loss_std

        mask = torch.abs(normalized_output) < upper_z
        temp = mask[0]
        for i in range(len(output[:])):
            temp = torch.logical_and(temp, mask[i])

        expanded_mask = temp.unsqueeze(1).expand(-1, module.shape[1])

        positive_temp = torch.logical_and(expanded_mask, negative_weight_mask)
        negative_temp = torch.logical_and(expanded_mask, positive_weight_mask)

        temp = torch.logical_or(positive_temp, negative_temp)
        current_weight[temp] = 0
        all_zeros = ~temp.any(dim=1)
        current_bias[all_zeros] = 0

        module.set_parameters(current_weight, current_bias)

    def recover(self, module, original_output, output):
        # Get the shapes and original parameters (weights and biases) of the module

        current_weight, current_bias = module.weight, module.bias
        original_weight, original_bias = module.get_parameters()
        normalized_output = output.clone()

        positive_weight_mask = original_weight > 0
        negative_weight_mask = original_weight < 0
        upper_z = norm.ppf(0.55)

        for b in range(len(output[:])):
            single_output = output[b]
            output_loss = original_output[b] - single_output[b]
            positive_loss_mask = output_loss > 0
            negative_loss_mask = output_loss < 0
            positive_loss_mean = torch.mean(single_output[positive_loss_mask])
            negative_loss_mean = torch.mean(single_output[negative_loss_mask])

            positive_loss_std = safe_std(single_output[positive_loss_mask], dim=0, epsilon=1e-5)
            negative_loss_std = safe_std(single_output[negative_loss_mask], dim=0, epsilon=1e-5)

            normalized_output[b][positive_loss_mask] = (single_output[positive_loss_mask] - positive_loss_mean)/positive_loss_std
            normalized_output[b][negative_loss_mask] = (single_output[negative_loss_mask] - negative_loss_mean)/negative_loss_std

        mask = torch.abs(normalized_output) < upper_z
        temp = mask[0]
        for i in range(len(output[:])):
            temp = torch.logical_and(temp, mask[i])

        expanded_mask = temp.unsqueeze(1).expand(-1, module.shape[1])

        positive_temp = torch.logical_and(expanded_mask, positive_weight_mask)
        negative_temp = torch.logical_and(expanded_mask, negative_weight_mask)

        temp = torch.logical_or(positive_temp, negative_temp)
        not_all_zeros = temp.any(dim=1)
        current_weight[temp] = original_weight[temp]
        current_bias[not_all_zeros] = original_bias[not_all_zeros]

        module.set_parameters(current_weight, current_bias)

    def propagate(self, module, input_tensor, positive_sample=True):
        # propagate input tensor to the module
        self.positive_sample = positive_sample
        output_tensor = self.propagate_embeddings(module.embeddings, input_tensor)
        output_tensor = self.propagate_encoder(
            module.encoder, output_tensor
        )
        output_tensor = self.propagate_pooler(module.pooler, output_tensor)
        output_tensor = module.dropout(output_tensor)
        output_tensor = self.propagate_classifier(module.classifier, output_tensor)
        return output_tensor

    def propagate_embeddings(self, module, input_tensor):
        output_tensor = module(input_tensor)
        return output_tensor

    def propagate_encoder(self, module, input_tensor):
        for i, encoder_block in enumerate(module.encoder_blocks):
            block_outputs = self.propagate_encoder_block(
                encoder_block, input_tensor, i, None
            )
            input_tensor = block_outputs[0]
        return input_tensor

    def propagate_encoder_block(
        self, module, input_tensor, i, head_mask=None
    ):
        def ff1_hook(module, input, output):
            if self.positive_sample:
                current_weight, current_bias = module.weight, module.bias
                original_outputs = module.layer(input[0])

                if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                    self.remove(module, output[:, 0, :])
                    pass
                else:
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.5:
                        # self.recover(module, original_outputs[:, 0, :], output[:, 0, :])
                        pass
                    else:
                        self.pruning(module, original_outputs[:, 0, :], output[:, 0, :])
                    pass
            else:
                current_weight, current_bias = module.weight, module.bias
                original_outputs = module.layer(input[0])

                if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                    self.remove(module, output[:, 0, :])
                    pass
                else:
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.5:
                        # self.recover(module, original_outputs[:, 0, :], output[:, 0, :])
                        pass
                    else:
                        self.pruning(module, original_outputs[:, 0, :], output[:, 0, :])
                    pass
            pass

        def ff2_hook(module, input, output):
            if self.positive_sample:
                current_weight, current_bias = module.weight, module.bias
                original_outputs = module.layer(input[0])

                if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                    self.remove(module, output[:, 0, :])
                    pass
                else:
                    pass
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.3:
                        # self.recover(module, original_outputs[:, 0, :], output[:, 0, :])
                        pass
                    else:
                        self.pruning(module, original_outputs[:, 0, :], output[:, 0, :])
                    pass
            else:
                current_weight, current_bias = module.weight, module.bias
                original_outputs = module.layer(input[0])

                if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                    self.remove(module, output[:, 0, :])
                    pass
                else:
                    pass
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.3:
                        # self.recover(module, original_outputs[:, 0, :], output[:, 0, :])
                        pass
                    else:
                        self.pruning(module, original_outputs[:, 0, :], output[:, 0, :])
                    pass

        attn_outputs = module.attention(input_tensor, None, None)
        handle = module.feed_forward1.dense.register_forward_hook(ff1_hook)
        intermediate_output = module.feed_forward1(attn_outputs)
        handle.remove()
        handle = module.feed_forward2.dense.register_forward_hook(ff2_hook)
        layer_output = module.feed_forward2(intermediate_output, attn_outputs)
        handle.remove()
        return (layer_output,)

    def propagate_attention_module(
        self, module, input_tensor, attention_mask, head_mask
    ):
        # handle = module.self_attention.register_forward_hook(attention_hook)
        self_outputs = module.self_attention(input_tensor, attention_mask, head_mask)
        # handle.remove()
        # handle = module.output.register_forward_hook(output_hook)
        attention_output = module(self_outputs[0], input_tensor)
        # handle.remove()
        return attention_output

    def propagate_pooler(self, module, input_tensor):
        first_token_tensor = input_tensor[:, 0]
        def pooler_hook(module, input, output):
            # Get the original output from model
            if self.positive_sample:
                current_weight, current_bias = module.weight, module.bias
                original_outputs = module.layer(input[0])

                if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                    self.remove(module, output)
                    pass
                else:
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.3:
                        # self.recover(module, original_outputs, output)
                        pass
                    else:
                        self.pruning(module, original_outputs, output)
                    pass
            else:
                current_weight, current_bias = module.weight, module.bias
                original_outputs = module.layer(input[0])

                if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                    self.remove(module, output)
                    pass
                else:
                    if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.3:
                        # self.recover(module, original_outputs, output)
                        pass
                    else:
                        self.pruning(module, original_outputs, output)
                    pass
            pass

        handle = module.dense.register_forward_hook(pooler_hook)
        output_tensor = module.dense(first_token_tensor)
        handle.remove()
        output_tensor = module.activation(output_tensor)
        return output_tensor

    def propagate_classifier(self, module, input_tensor):

        output_tensor = module(input_tensor)
        return output_tensor


    def remove_broken_weights(self, module):
        pass

    def recover_broken_weights(self, module):
        pass
