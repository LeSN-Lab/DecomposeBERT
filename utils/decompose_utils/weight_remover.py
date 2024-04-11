import torch
from utils.type_utils.data_type import safe_std
from utils.model_utils.modular_layers import set_parameters
from scipy.stats import norm


class WeightRemoverBert:
    def __init__(self, model, p=0.9):
        self.source_model = model
        self.p = p

    def propagate(self, module, input_tensor):
        # propagate input tensor to the module
        output_tensor = self.propagate_embeddings(
            self.source_model.bert.embeddings, module.bert.embeddings, input_tensor
        )
        output_tensor = self.propagate_encoder(
            self.source_model.bert.encoder, module.bert.encoder, output_tensor
        )
        output_tensor = self.propagate_pooler(
            self.source_model.bert.pooler, module.bert.pooler, output_tensor
        )
        output_tensor = module.dropout(output_tensor)
        output_tensor = self.propagate_classifier(
            self.source_model.classifier, module.classifier, output_tensor
        )
        return output_tensor

    def propagate_embeddings(self, ref_model, module, input_tensor):
        output_tensor = module(input_tensor)
        return output_tensor

    def propagate_encoder(self, ref_model, module, input_tensor):
        for i, encoder_block in enumerate(module.layer):
            block_outputs = self.propagate_encoder_block(
                ref_model.layer[i], encoder_block, input_tensor
            )
            input_tensor = block_outputs[0]
        return input_tensor

    def propagate_encoder_block(self, ref_model, module, input_tensor):
        def ff1_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias
            ref = ref_model.intermediate.dense
            original_outputs = ref(input[0])

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.remove(module, output[:, 0, :])

        def ff2_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias
            ref = ref_model.output.dense
            original_outputs = ref(input[0])

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.remove(module, output[:, 0, :])

        attn_outputs = module.attention(input_tensor, None, None)
        handle = module.intermediate.dense.register_forward_hook(ff1_hook)
        intermediate_output = module.intermediate(attn_outputs[0])
        handle.remove()
        handle = module.output.dense.register_forward_hook(ff2_hook)
        layer_output = module.output(intermediate_output, attn_outputs[0])
        handle.remove()
        return (layer_output,)

    def propagate_attention_module(
        self, ref_model, module, input_tensor, attention_mask, head_mask
    ):
        # handle = module.self_attention.register_forward_hook(attention_hook)
        self_outputs = module.self_attention(input_tensor, attention_mask, head_mask)
        # handle.remove()
        # handle = module.output.register_forward_hook(output_hook)
        attention_output = module(self_outputs[0], input_tensor)
        # handle.remove()
        return attention_output

    def propagate_pooler(self, ref_model, module, input_tensor):
        first_token_tensor = input_tensor[:, 0]

        def pooler_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias
            ref = ref_model.dense
            original_outputs = ref(input[0])

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.remove(module, output)

        handle = module.dense.register_forward_hook(pooler_hook)
        output_tensor = module.dense(first_token_tensor)
        handle.remove()
        output_tensor = module.activation(output_tensor)
        return output_tensor

    def propagate_classifier(self, ref_model, module, input_tensor):
        def classifier_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias
            ref = ref_model.dense
            original_outputs = ref(input[0])

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.remove(module, output)
        # handle = module.register_forward_hook(classifier_hook)
        output_tensor = module(input_tensor)
        # handle.remove()
        return output_tensor

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
        shape = current_weight.shape

        output_mean = torch.mean(output, dim=1, keepdim=True)
        output_std = torch.std(output, dim=1, keepdim=True)
        z_scores = (output - output_mean) / output_std

        lower_z, upper_z = norm.ppf(0.45), norm.ppf(0.55)

        mask = torch.logical_and(z_scores >= lower_z, z_scores < upper_z)
        mask = torch.all(mask, dim=0).unsqueeze(1).expand(-1, shape[1])

        current_weight[mask] = 0
        all_zeros = ~mask.any(dim=1)
        current_bias[all_zeros] = 0
        set_parameters(module, current_weight, current_bias)
