import torch
from utils.model_utils.modular_layers import set_parameters
class ConcernModularizationBert:

    @staticmethod
    def channeling(module, concern, device):
        current_weight, current_bias = module.weight.clone(), module.bias.clone()
        mask = torch.tensor([False] * current_weight.size(0), dtype=torch.bool)
        mask[concern] = True
        true_concern_weights = current_weight[mask, :]
        true_concern_bias = current_bias[mask]

        false_concern_weights = torch.zeros(module.shape[1]).unsqueeze(dim=0).to(device)
        for j in range(module.shape[1]):
            false_concern_weights[0, j] = torch.mean(current_weight[~mask, j])

        false_concern_bias = torch.mean(current_bias[~mask], dim=0, keepdim=True)
        new_weight = torch.cat([true_concern_weights, false_concern_weights], dim=0)
        new_bias = torch.cat([true_concern_bias, false_concern_bias], dim=0)

        set_parameters(module, new_weight, new_bias)