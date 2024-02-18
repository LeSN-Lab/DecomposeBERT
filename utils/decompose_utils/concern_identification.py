import torch.nn as nn
import torch

class ConcernIdentificationBert:
    def __init__(self, config):
        self.config = config

    def propagate(self, module, input_tensor):
        pass

    def get_activation(self, module, input_tensor):
        pass


def print_active_nodes_count(module, input, output):

    node_count = torch.sum(output > 0).item()
    print(f"{module.name}0 이상인 노드 개수: {node_count}")


def break_edge():
    pass

def restore_edge():
    pass