import torch.nn as nn
import torch


class ConcernIdentificationBert:
    def __init__(self, config):
        self.config = config
        self.activations = []

    def hook_fn(self, module, input, output):
        # 활성화된 노드의 위치를 기록합니다. 여기서는 단순히 값이 0보다 큰 노드를 활성화된 것으로 가정합니다.
        activated_nodes = output > 0
        self.activations.append({module.name: activated_nodes.float()})

    def register_hook(self, module):
        # 모듈에 대한 후크를 등록합니다.
        module.register_forward_hook(self.hook_fn)

    def propagate(self, module, input_tensor):
        # 모델에 입력을 전달하고, 활성화된 노드의 위치를 반환합니다.
        output_tensor = module(input_tensor)
        return output_tensor

    def analyze_activations(self):
        # 활성화된 노드의 분포를 분석합니다. 예를 들어, 평균 활성화율을 계산할 수 있습니다.
        mean_activation = torch.mean(torch.cat(self.activations, dim=0), dim=0)
        return mean_activation


def print_active_nodes_count(module, input, output):

    node_count = torch.sum(output > 0).item()
    print(f"{module.name}0 이상인 노드 개수: {node_count}")

