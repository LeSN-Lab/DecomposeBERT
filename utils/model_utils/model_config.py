from utils.type_utils.architecture_type import ArchitectureType

import os
from utils.paths import p


class ModelConfig:
    def __init__(
        self,
        _model_name,
        _model_dir,
        _data,
        _is_pretrained,
        _transformer_config,
        _checkpoint_name=None,
        _device="cuda:0",
    ):
        # Specific directories
        t = os.path.join(_model_dir, _model_name)
        self.is_downloaded = p.is_dir(os.path.join(p.Config, t))
        self.model_dir = p.get_dir(p.Config, t)
        self.train_dir = p.get_dir(p.Train, t)
        self.data_dir = p.get_dir(p.Data, _data)

        # others
        self.model_name = _model_name
        self.data = _data
        self.is_pretrained = _is_pretrained
        self.transformer_config = _transformer_config
        self.arg_type = self.get_architecture(_model_name)
        self.checkpoint_name = _checkpoint_name
        self.device = _device

    @staticmethod
    def get_architecture(model_name):
        if "bert" in model_name.lower():
            return ArchitectureType.only_encoder
        elif "gpt" in model_name.lower():
            return ArchitectureType.only_decoder
        else:
            return ArchitectureType.both
