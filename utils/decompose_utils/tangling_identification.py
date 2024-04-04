import torch
from utils.type_utils.data_type import safe_std
from scipy.stats import norm

class TanglingIdentification:
    def __init__(self, model_config):
        self.positive_sample = True

    def recover(self, module, original_output, output):
        # Get the shapes and original parameters (weights and biases) of the module

        current_weight, current_bias = module.weight.clone(), module.bias.clone()
        original_weight, original_bias = module.get_parameters()

        positive_weight_mask = original_weight > 0
        negative_weight_mask = original_weight < 0
        upper_z = norm.ppf(0.8)
        output_loss = original_output - output

        positive_loss_mask = output_loss > 0
        negative_loss_mask = output_loss < 0

        common_positive_mask = torch.all(positive_loss_mask, dim=0).unsqueeze(1).expand(-1, module.shape[1])
        common_negative_mask = torch.all(negative_loss_mask, dim=0).unsqueeze(1).expand(-1, module.shape[1])

        weight_difference = original_weight != current_weight
        not_included_positive_weight = torch.logical_and(weight_difference, positive_weight_mask)
        not_included_negative_weight = torch.logical_and(weight_difference, negative_weight_mask)

        filtered_positive_weight = torch.where(not_included_positive_weight, original_weight, torch.tensor(0.0))
        filtered_negative_weight = torch.where(not_included_negative_weight, original_weight, torch.tensor(0.0))
        positive_weight_mean = torch.mean(filtered_positive_weight, dim=1, keepdim=True)
        positive_weight_std = safe_std(filtered_negative_weight, dim=1, epsilon=1e-5, keepdim=True)

        negative_weight_mean = torch.mean(filtered_negative_weight, dim=1, keepdim=True)
        negative_weight_std = safe_std(filtered_negative_weight, dim=1, epsilon=1e-5)
        positive_z_scores = (original_weight - positive_weight_mean) / positive_weight_std
        negative_z_scores = (original_weight - negative_weight_mean) / negative_weight_std

        positive_recovery_mask = torch.logical_and(positive_z_scores <= upper_z, positive_z_scores > 0)
        mask = torch.logical_and(common_positive_mask, positive_recovery_mask)
        current_weight[mask] = original_weight[mask]

        negative_recovery_mask = torch.logical_and(negative_z_scores >= -upper_z, negative_z_scores < 0)
        mask = torch.logical_and(common_negative_mask, negative_recovery_mask)
        current_weight[mask] = original_weight[mask]

        not_all_zeros = current_weight.any(dim=1)
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
            current_weight, current_bias = module.weight, module.bias
            original_output = module.layer(input[0])
            self.recover(module, original_output[:, 0, :], output[:, 0, :])

            # if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.60:
            #     self.recover(module, original_output[:, 0, :], output[:, 0, :])
            #     pass
            # else:
            #     # self.remove(module, output[:, 0, :])
            #     pass

        def ff2_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias
            original_output = module.layer(input[0])
            self.recover(module, original_output[:, 0, :], output[:, 0, :])

            # if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.40:
            #     self.recover(module, original_output[:, 0, :], output[:, 0, :])
            #     pass
            # else:
            #     # self.remove(module, output[:, 0, :])
            #     pass

        attn_outputs = module.attention(input_tensor, None, None)
        self.attn_probs = module.attention.self_attention.attention_probs
        handle = module.feed_forward1.dense.register_forward_hook(ff1_hook)
        intermediate_output = module.feed_forward1(attn_outputs)
        handle.remove()
        handle = module.feed_forward2.dense.register_forward_hook(ff2_hook)
        layer_output = module.feed_forward2(intermediate_output, attn_outputs)
        handle.remove()
        return (layer_output,)

    def propagate_attention_module(
        self, module, input_tensor, head_mask
    ):
        # handle = module.self_attention.register_forward_hook(attention_hook)
        self_outputs = module.self_attention(input_tensor, head_mask)
        # handle.remove()
        # handle = module.output.register_forward_hook(output_hook)
        attention_output = module(self_outputs[0], input_tensor)
        # handle.remove()
        return attention_output

    def propagate_pooler(self, module, input_tensor):
        first_token_tensor = input_tensor[:, 0]
        def pooler_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias
            original_outputs = module.layer(input[0])
            self.recover(module, original_outputs[0], output)

            # if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.40:
            #     self.recover(module, original_outputs[0], output)
            #     pass
            # else:
            #     # self.remove(module, output)
            #     pass

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
