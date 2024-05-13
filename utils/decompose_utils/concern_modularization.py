import torch
from utils.model_utils.modular_layers import set_parameters
class ConcernModularizationBert:
    @staticmethod
    def channeling(module, active_node, dead_node, concern_idx, device):
        weight = module.classifier.weight
        bias = module.classifier.bias
        active_top1 = max(active_node)
        dead_top1 = max(dead_node)
        active = [idx for idx, val in enumerate(active_node) if val >= active_top1/2]
        dead = [idx for idx, val in enumerate(dead_node) if val >= dead_top1 / 2]

        print(active)
        print(dead)

        inter1 = [1 if idx in dead else 0 for idx, val in enumerate(dead_node)]
        inter2 = [1 if idx in active and idx != concern_idx else 0 for idx, val in enumerate(active_node)]

        inter = torch.asarray([inter1, inter2], dtype=torch.float32).to(device)
        norms = inter.norm(p=1, dim=1, keepdim=True)
        inter_normalized = inter / norms
        inter_normalized[0] = inter_normalized[0]
        inter_normalized[1] = inter_normalized[1]
        new_weight = torch.matmul(inter_normalized, weight)
        new_bias = torch.matmul(inter_normalized, bias)

        set_parameters(module.classifier, new_weight, new_bias)
        module.classifier.out_features = 2