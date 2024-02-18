import torch.nn as nn


class ConcernIdentificationBert:
    def __init__(self, config):
        self.config = config

    def propagate(self, module, input_tensor):
        pass

    def get_activation(self, module, input_tensor):
        def hook(moodule, input, output):
            