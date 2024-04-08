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
        current_weight, current_bias = (
            module.weight.clone(),
            module.bias.clone(),
        )  # updating parameters

        percentile_40 = torch.quantile(output, 0.4, dim=1, keepdim=True)
        percentile_60 = torch.quantile(output, 0.6, dim=1, keepdim=True)
        mask = torch.logical_and(output >= percentile_40, output < percentile_60)
        mask = torch.all(mask, dim=0).unsqueeze(1).expand(-1, module.shape[1])

        current_weight[mask] = 0
        all_zeros = ~mask.any(dim=1)
        current_bias[all_zeros] = 0
        module.set_parameters(current_weight, current_bias)

    def pruning(self, module, original_output, output):
        # Get the shapes and original parameters (weights and biases) of the module

        current_weight, current_bias = module.weight.clone(), module.bias.clone()
        original_weight, original_bias = module.get_parameters()

        original_weight_mean = torch.nanmean(original_weight, dim=1, keepdim=True)
        original_weight_std = safe_std(original_weight, dim=1, keepdim=True)

        current_weight_std = safe_std(
            current_weight,
            epsilon=original_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )
        output_loss = original_output - output
        positive_loss_mask = (
            torch.all(output_loss > 0, dim=0).unsqueeze(1).expand(-1, module.shape[1])
        )
        cgto_std_mask = current_weight_std > original_weight_std
        expanded_cgto_std_mask = cgto_std_mask.expand(-1, module.shape[1])

        padded_positive = torch.where(
            current_weight >= 0, current_weight, torch.tensor(float("nan"))
        )
        padded_negative = torch.where(
            current_weight <= 0, current_weight, torch.tensor(float("nan"))
        )

        top_positive_quantile = (
            torch.nanquantile(padded_positive, 0.75, dim=1)
            .unsqueeze(1)
            .expand(-1, module.shape[1]),
        )

        bottom_positive_quantile = (
            torch.nanquantile(
                padded_positive,
                0.25,
                dim=1,
            )
            .unsqueeze(1)
            .expand(-1, module.shape[1])
        )

        top_negative_quantile = (
            torch.nanquantile(
                padded_negative,
                0.75,
                dim=1,
            )
            .unsqueeze(1)
            .expand(-1, module.shape[1])
        )
        bottom_negative_quantile = (
            torch.nanquantile(padded_negative, 0.25, dim=1)
            .unsqueeze(1)
            .expand(-1, module.shape[1])
        )

        outlier_remove_mask = torch.where(
            ~positive_loss_mask,
            torch.logical_or(
                padded_negative < bottom_negative_quantile[0],
                padded_negative >= top_negative_quantile[0],
            ),
            torch.logical_or(
                padded_positive >= top_positive_quantile[0],
                padded_positive < bottom_positive_quantile[0],
            ),
        )
        outlier_remove_mask = torch.logical_and(
            outlier_remove_mask, expanded_cgto_std_mask
        )

        top_positive_quantile = (
            torch.nanquantile(padded_positive, 0.55, dim=1)
            .unsqueeze(1)
            .expand(-1, module.shape[1]),
        )

        bottom_positive_quantile = (
            torch.nanquantile(
                padded_positive,
                0.45,
                dim=1,
            )
            .unsqueeze(1)
            .expand(-1, module.shape[1])
        )

        top_negative_quantile = (
            torch.nanquantile(
                padded_negative,
                0.55,
                dim=1,
            )
            .unsqueeze(1)
            .expand(-1, module.shape[1])
        )
        bottom_negative_quantile = (
            torch.nanquantile(padded_negative, 0.45, dim=1)
            .unsqueeze(1)
            .expand(-1, module.shape[1])
        )
        neutral_remove_mask = torch.where(
            ~positive_loss_mask,
            torch.logical_and(
                padded_negative >= bottom_negative_quantile[0],
                padded_negative < top_negative_quantile[0],
            ),
            torch.logical_and(
                padded_positive < top_positive_quantile[0],
                padded_positive >= bottom_positive_quantile[0],
            ),
        )
        neutral_remove_mask = torch.logical_and(
            neutral_remove_mask, ~expanded_cgto_std_mask
        )

        remove_mask = torch.logical_or(outlier_remove_mask, neutral_remove_mask)
        current_weight[remove_mask] = 0

        all_zeros = ~current_weight.any(dim=1)
        current_bias[all_zeros] = 0

        module.set_parameters(current_weight, current_bias)

    def propagate(self, module, input_tensor, positive_sample=True):
        # propagate input tensor to the module
        self.positive_sample = positive_sample
        output_tensor = self.propagate_embeddings(module.embeddings, input_tensor)
        output_tensor = self.propagate_encoder(module.encoder, output_tensor)
        output_tensor = self.propagate_pooler(module.pooler, output_tensor)
        output_tensor = module.dropout(output_tensor)
        output_tensor = self.propagate_classifier(module.classifier, output_tensor)
        return output_tensor

    def propagate_embeddings(self, module, input_tensor):
        output_tensor = module(input_tensor)
        return output_tensor

    def propagate_encoder(self, module, input_tensor):
        for i, encoder_block in enumerate(module.encoder_blocks):
            block_outputs = self.propagate_encoder_block(encoder_block, input_tensor)
            input_tensor = block_outputs[0]
        return input_tensor

    def propagate_encoder_block(self, module, input_tensor):
        def ff1_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias
            original_outputs = module.layer(input[0])

            if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                self.remove(module, output[:, 0, :])
            elif torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.4:
                self.pruning(module, original_outputs[:, 0, :], output[:, 0, :])
                pass

        def ff2_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias
            original_outputs = module.layer(input[0])
            if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                self.remove(module, output[:, 0, :])
            elif torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.4:
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
            current_weight, current_bias = module.weight, module.bias
            original_outputs = module.layer(input[0])

            if torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.9:
                self.remove(module, output)
            elif torch.sum(current_weight != 0) > module.shape[0] * module.shape[1] * 0.4:
                self.pruning(module, original_outputs, output)
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
