import torch
class ConcernModularizationBert:

    @staticmethod
    def channeling(module1, concern, device):
        original_weight, original_bias = module1.get_parameters()
        mask = torch.tensor([False] * original_weight.size(0), dtype=torch.bool)
        mask[concern] = True
        true_concern_weights = original_weight[mask,:]
        true_concern_bias = original_bias[mask]

        false_concern_weights = torch.zeros(module1.shape[1]).unsqueeze(dim=0).to(device)
        for j in range(module1.shape[1]):
            false_concern_weights[0, j] = torch.mean(original_weight[~mask, j])

        false_concern_bias = torch.mean(original_bias[~mask], dim=0, keepdim=True)
        new_weight = torch.cat([true_concern_weights, false_concern_weights], dim=0)
        new_bias = torch.cat([true_concern_bias, false_concern_bias], dim=0)

        module1.set_parameters(new_weight, new_bias)