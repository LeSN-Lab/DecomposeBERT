import torch
from utils.decompose_utils.calc_util import safe_std
from utils.model_utils.modular_layers import set_parameters
from scipy.stats import norm
import torch.nn.functional as F


class TanglingIdentification:
    def __init__(self, model, p=0.7):
        self.source_model = model
        self.p = p
        self.dead_node = [0] * model.classifier.weight.shape[0]

    def propagate(self, module, original_input_tensor):
        # propagate input tensor to the module
        original_output_tensor, current_output_tensor = self.propagate_embeddings(
            self.source_model.bert.embeddings,
            module.bert.embeddings,
            original_input_tensor.clone(),
            original_input_tensor.clone(),
        )
        original_output_tensor, current_output_tensor = self.propagate_encoder(
            self.source_model.bert.encoder,
            module.bert.encoder,
            original_output_tensor,
            current_output_tensor,
        )
        original_output_tensor, current_output_tensor = self.propagate_pooler(
            self.source_model.bert.pooler,
            module.bert.pooler,
            original_output_tensor,
            current_output_tensor,
        )
        original_output_tensor, current_output_tensor = self.propagate_classifier(
            self.source_model.classifier,
            module.classifier,
            original_output_tensor,
            current_output_tensor,
        )
        return original_output_tensor, current_output_tensor

    def propagate_embeddings(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_output_tensor = ref_model(original_input_tensor)
        current_output_tensor = module(current_input_tensor)
        return original_output_tensor, current_output_tensor

    def propagate_encoder(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        for i, encoder_block in enumerate(module.layer):
            original_output_tensor, current_output_tensor = (
                self.propagate_encoder_block(
                    ref_model.layer[i],
                    encoder_block,
                    original_input_tensor,
                    current_input_tensor,
                )
            )
            original_input_tensor = original_output_tensor[0]
            current_input_tensor = current_output_tensor[0]
        return original_input_tensor, current_input_tensor

    def propagate_encoder_block(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_attn_output = ref_model.attention(original_input_tensor, None, None)
        current_attn_output = module.attention(current_input_tensor, None, None)
        original_output_tensor = ref_model.intermediate.dense(original_attn_output[0])

        def ff1_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) < torch.numel(current_weight) * self.p:
                self.recover(
                    ref_model.intermediate.dense,
                    module,
                    original_output_tensor[:, 0, :],
                    output[:, 0, :],
                )

        handle = module.intermediate.dense.register_forward_hook(ff1_hook)
        module.intermediate.dense(current_attn_output[0])
        handle.remove()

        original_intermediate_output = ref_model.intermediate(original_attn_output[0])
        current_intermediate_output = module.intermediate(current_attn_output[0])

        original_output_tensor = ref_model.output.dense(original_intermediate_output)

        def ff2_hook(module, input, output):
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) < torch.numel(current_weight) * self.p:
                self.recover(
                    ref_model.output.dense,
                    module,
                    original_output_tensor[:, 0, :],
                    output[:, 0, :],
                )

        handle = module.output.dense.register_forward_hook(ff2_hook)
        module.output.dense(current_intermediate_output)
        handle.remove()
        original_output = ref_model.output(
            original_intermediate_output, original_attn_output[0]
        )
        current_output = module.output(
            current_intermediate_output, current_attn_output[0]
        )

        return (original_output,), (current_output,)

    def propagate_pooler(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_first_token_tensor = original_input_tensor[:, 0]
        current_first_token_tensor = current_input_tensor[:, 0]
        original_output_tensor = ref_model.dense(original_first_token_tensor)

        def pooler_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias

            if torch.sum(current_weight != 0) < torch.numel(current_weight) * self.p:
                self.recover(ref_model.dense, module, original_output_tensor, output)

        handle = module.dense.register_forward_hook(pooler_hook)
        module.dense(current_first_token_tensor)
        handle.remove()
        original_output_tensor = ref_model(original_input_tensor)
        current_output_tensor = module(current_input_tensor)

        return original_output_tensor, current_output_tensor

    def propagate_classifier(
        self, ref_model, module, original_input_tensor, current_input_tensor
    ):
        original_output_tensor = ref_model(original_input_tensor)

        def classifier_hook(module, input, output):
            # Get the original output from model
            current_weight, current_bias = module.weight, module.bias
            temp_mean = output.mean(dim=1, keepdim=True)
            temp_std = output.std(dim=1, keepdim=True)
            temp_score = (output - temp_mean) / (temp_std + 1e-5)
            temp = torch.any((temp_score > 0), dim=0).tolist()
            self.dead_node = [a + b for a, b in zip(temp, self.dead_node)]

            # if torch.sum(current_weight != 0) < torch.numel(current_weight) * self.p:
            #     self.recover(ref_model, module, original_output_tensor, output)

        handle = module.register_forward_hook(classifier_hook)
        current_output_tensor = module(current_input_tensor)
        handle.remove()
        return original_output_tensor, current_output_tensor

    def recover(self, ref_model, module, original_output, output):
        # Get the shapes and original parameters (weights and biases) of the module

        current_weight, current_bias = module.weight.clone(), module.bias.clone()
        original_weight, original_bias = (
            ref_model.weight.clone(),
            ref_model.bias.clone(),
        )
        shape = current_weight.shape

        original_weight_std = safe_std(original_weight, dim=1, keepdim=True)
        current_weight_std = safe_std(
            current_weight,
            epsilon=original_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )
        cgto_std_mask = current_weight_std > original_weight_std
        expanded_cgto_std_mask = cgto_std_mask.expand(-1, shape[1])

        not_included = torch.where(
            original_weight != current_weight,
            original_weight,
            torch.tensor(float("nan")),
        )
        not_included_mean = torch.nanmean(not_included, dim=1, keepdim=True)
        not_included_std = safe_std(
            not_included,
            epsilon=original_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )
        z_scores = torch.where(
            ~torch.isnan(not_included),
            (not_included - not_included_mean) / not_included_std,
            torch.tensor(float("nan")),
        )

        neutral_lower_z, neutral_upper_z = norm.ppf(0.45), norm.ppf(0.55)
        outlier_lower_z, outlier_upper_z = norm.ppf(0.05), norm.ppf(0.95)

        neutral_recovery_mask = torch.logical_and(
            z_scores >= neutral_lower_z, z_scores < neutral_upper_z
        )
        outlier_recovery_mask = torch.logical_or(
            z_scores < outlier_lower_z, z_scores >= outlier_upper_z
        )

        recovery_mask = torch.where(
            expanded_cgto_std_mask, neutral_recovery_mask, outlier_recovery_mask
        )

        current_weight[recovery_mask] = original_weight[recovery_mask]

        not_all_zeros = current_weight.any(dim=1)
        current_bias[not_all_zeros] = original_bias[not_all_zeros]
        set_parameters(module, current_weight, current_bias)
