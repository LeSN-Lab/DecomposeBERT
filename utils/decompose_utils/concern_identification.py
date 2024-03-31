import torch
from utils.model_utils.modular_layers import get_extended_attention_mask
from scipy.stats import norm

class ConcernIdentificationBert:
    def __init__(self, model_config):
        self.positive_sample = True
        self.is_sparse = {"ff1": [False]* model_config.num_hidden_layers, "ff2":[False]*model_config.num_hidden_layers, "pooler":False}

    def positive_hook(self, module, output, is_sparse):
        """
        Positive hook
        Attributes:
            module (Layer): custom layer
            output (torch.Tensor): output tensor of the original layer
        """
        # Get the shapes and original parameters (weights and biases) of the layer
        current_weight, current_bias = module.weight, module.bias   # updating parameters
        original_weight, original_bias = module.get_parameters()

        positive_output_mask = output[0] > 0
        negative_output_mask = output[0] < 0

        positive_weight_mask = original_weight > 0
        negative_weight_mask = original_weight < 0
        positive_extended_mask = positive_output_mask.unsqueeze(1).expand(-1, module.shape[1])
        negative_extended_mask = negative_output_mask.unsqueeze(1).expand(-1, module.shape[1])

        # if is_sparse:
        #     # recover positive_output
        #     temp = positive_extended_mask
        #     not_all_zeros = temp.any(dim=1)
        #     current_weight[temp] = original_weight[temp]
        #     current_bias[not_all_zeros] = original_bias[not_all_zeros]
        # else:
        if not is_sparse:
            upper_z = norm.ppf(0.70)
            mean_output = torch.mean(output[0])
            normalized_output = (output[0] - mean_output) / torch.std(output[0])
            temp = torch.abs(normalized_output) < upper_z
            expanded_temp = temp.unsqueeze(1).expand(-1, module.shape[1])
            temp = expanded_temp
            # temp = torch.logical_and(expanded_temp, negative_extended_mask)
            current_weight[temp] = 0
            all_zeros = ~temp.any(dim=1)
            current_bias[all_zeros] = 0

        module.set_parameters(current_weight, current_bias)

        if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.15:
            is_sparse = True
        else:
            is_sparse = False

        return is_sparse

    def negative_hook(self, module, original_output, output, is_sparse):
        # Get the shapes and original parameters (weights and biases) of the module
        current_weight, current_bias = module.weight, module.bias
        original_weight, original_bias = module.get_parameters()
        normalized_output = output.clone()

        positive_weight_mask = original_weight > 0
        negative_weight_mask = original_weight < 0

        output_loss = original_output - output
        positive_loss_mask = output_loss[0] > 0
        negative_loss_mask = output_loss[0] < 0

        positive_loss_mean = torch.mean(output[positive_loss_mask])
        negative_loss_mean = torch.mean(output[negative_loss_mask])

        positive_loss_std = torch.std(output[positive_loss_mean])
        negative_loss_std = torch.std(output[negative_loss_mask])

        normalized_output[positive_loss_mask] = (output[positive_loss_mask] - positive_loss_mean)/positive_loss_std
        normalized_output[negative_loss_mask] = (output[negative_loss_mask] - negative_loss_mean)/negative_loss_std

        upper_z = norm.ppf(0.95)
        temp = torch.abs(normalized_output) > upper_z
        expanded_temp = temp.unsqueeze(1).expand(-1, module.shape[1])

        positive_temp = torch.logitcal_and(torch.logical_and(expanded_temp, positive_loss_mask), negative_weight_mask)
        negative_temp = torch.logical_and(torch.logical_and(expanded_temp, negative_loss_mask), positive_weight_mask)

        temp = torch.logical_or(positive_temp, negative_temp)
        not_all_zeros = temp.any(dim=1)
        current_weight[temp] = original_weight[temp]
        current_bias[not_all_zeros] = original_bias[not_all_zeros]

        # else:
            # temp = torch.logical_and(positive_extended_mask, threshold_weight_mask)
            # temp = torch.logical_and(positive_extended_mask, threshold_weight_mask)
            # current_weight[temp] = 0
            # all_zeros = ~temp.any(dim=1)
            # current_bias[all_zeros] = 0

        module.set_parameters(current_weight, current_bias)

        if torch.sum(current_weight != 0) < module.shape[0] * module.shape[1] * 0.15:
            return True
        else:
            return False

    def propagate(self, module, input_tensor, attention_mask, positive_sample=True):
        # propagate input tensor to the module
        self.positive_sample = positive_sample
        output_tensor = self.propagate_embeddings(module.embeddings, input_tensor)
        output_tensor = self.propagate_encoder(
            module.encoder, output_tensor, attention_mask
        )
        output_tensor = self.propagate_pooler(module.pooler, output_tensor)
        output_tensor = module.dropout(output_tensor)
        output_tensor = self.propagate_classifier(module.classifier, output_tensor)
        return output_tensor

    def propagate_embeddings(self, module, input_tensor):
        output_tensor = module(input_tensor)
        return output_tensor

    def propagate_encoder(self, module, input_tensor, attention_mask):
        attention_mask = get_extended_attention_mask(attention_mask)
        for i, encoder_block in enumerate(module.encoder_blocks):
            block_outputs = self.propagate_encoder_block(
                encoder_block, input_tensor, attention_mask, i, None
            )
            input_tensor = block_outputs[0]
        return input_tensor

    def propagate_encoder_block(
        self, module, input_tensor, attention_mask, i, head_mask=None
    ):
        def ff1_hook(module, input, output):
            if self.positive_sample:
                # original_output = module.layer(input[0])
                self.is_sparse["ff1"][i] = self.positive_hook(module, output[:, 0, :], self.is_sparse["ff1"][i])
            else:
                original_output = module.layer(input[0])
                self.is_sparse["ff1"][i] = self.negative_hook(module, original_output[:, 0, :], output[:, 0, :], self.is_sparse["ff1"][i])

        def ff2_hook(module, input, output):
            if self.positive_sample:
                # original_output = module.layer(input[0])
                self.is_sparse["ff2"][i] = self.positive_hook(module, output[:, 0, :], self.is_sparse["ff2"][i])
            else:
                original_output = module.layer(input[0])
                self.is_sparse["ff2"][i] = self.negative_hook(module, original_output[:, 0, :], output[:, 0, :], self.is_sparse["ff2"][i])

        attn_outputs = module.attention(input_tensor, attention_mask, head_mask)
        self.attn_probs = module.attention.self_attention.attention_probs
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
                # original_outputs = module.layer(input[0])
                self.is_sparse["pooler"] = self.positive_hook(module, output, self.is_sparse["pooler"])
            else:
                original_outputs = module.layer(input[0])
                self.is_sparse["pooler"] = self.negative_hook(module, original_outputs, output, self.is_sparse["pooler"])

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
