import torch
from utils.model_utils.modular_layers import set_parameters
class ConcernModularizationBert:
    @staticmethod
    def channeling(module, active_node, dead_node, concern_idx, device):
        weight = module.classifier.weight
        bias = module.classifier.bias
        inter1 = [dead_node[i] if i != concern_idx and dead_node[i] else 0 for i in range(len(dead_node))]
        inter2 = [active_node[i] if i == concern_idx else 0 for i in range(len(active_node))]
        # inter1 = dead_node
        # inter2 = active_node

        inter = torch.asarray([inter1, inter2], dtype=torch.float32).to(device)
        norms = inter.norm(p=1, dim=1, keepdim=True)
        inter_normalized = inter / norms
        inter_normalized[0] = inter_normalized[0]
        inter_normalized[1] = inter_normalized[1]
        new_weight = torch.matmul(inter_normalized, weight)
        new_bias = torch.matmul(inter_normalized, bias)

        set_parameters(module.classifier, new_weight, new_bias)
        module.classifier.out_features = 2