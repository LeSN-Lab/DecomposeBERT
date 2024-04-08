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

        positive_weights = torch.where(current_weight >= 0, current_weight, torch.nan)
        negative_weights = torch.where(current_weight <= 0, current_weight, torch.nan)

        original_weight_std = safe_std(original_weight, dim=1, keepdim=True)
        current_weight_std = safe_std(current_weight, epsilon=original_weight_std, unbiased=True, dim=1, keepdim=True)

        positive_weight_mean = torch.nanmean(positive_weights, dim=1, keepdim=True)
        positive_weight_std = safe_std(positive_weights, epsilon=original_weight_std, unbiased=True, dim=1, keepdim=True)
        positive_z_scores = torch.where(torch.isnan(positive_weights), -positive_weight_mean/positive_weight_std,
                                        (positive_weights - positive_weight_mean) / positive_weight_std)

        negative_weight_mean = torch.nanmean(negative_weights, dim=1, keepdim=True)
        negative_weight_std = safe_std(negative_weights, epsilon=original_weight_std, unbiased=True, dim=1, keepdim=True)
        negative_z_scores = torch.where(torch.isnan(negative_weights), -negative_weight_mean/negative_weight_std,
                                        (negative_weights - negative_weight_mean) / negative_weight_std)

        cgto_std_mask = current_weight_std > original_weight_std
        expanded_cgto_std_mask = cgto_std_mask.expand(-1, module.shape[1])

        output_loss = original_output - output
        positive_loss_mask = torch.all(output_loss > 0, dim=0).unsqueeze(1).expand(-1, module.shape[1])
        positive_recovery_mask = positive_z_scores >= torch.quantile(positive_z_scores, 0.75, dim=1, keepdim=True)
        negative_recovery_mask = negative_z_scores <= torch.quantile(negative_z_scores, 0.25, dim=1, keepdim=True)

        po = torch.logical_and(positive_z_scores <= torch.quantile(positive_z_scores, 0.25, dim=1, keepdim=True), original_weight >= 0)
        no = torch.logical_and(negative_z_scores >= torch.quantile(negative_z_scores, 0.75, dim=1, keepdim=True), original_weight <= 0)
        neutral_recovery_mask = torch.logical_or(po, no)

        # outlier_mask = torch.where(positive_loss_mask, negative_recovery_mask, positive_recovery_mask)
        outlier_mask = torch.where(positive_loss_mask, positive_recovery_mask, negative_recovery_mask)
        # recovery_mask = torch.where(expanded_cgto_std_mask, neutral_recovery_mask, outlier_mask)
        recovery_mask = torch.where(expanded_cgto_std_mask, outlier_mask, neutral_recovery_mask)

        current_weight[recovery_mask] = original_weight[recovery_mask]

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

            if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.70:
                self.recover(module, original_output[:, 0, :], output[:, 0, :])
                pass
            else:
                # self.remove(module, output[:, 0, :])
                pass

        def ff2_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias
            original_output = module.layer(input[0])

            if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.40:
                self.recover(module, original_output[:, 0, :], output[:, 0, :])
                pass
            else:
                # self.remove(module, output[:, 0, :])
                pass

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

            if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.40:
                self.recover(module, original_outputs[0], output)
                pass
            else:
                # self.remove(module, output)
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
