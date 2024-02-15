from utils.type_utils.architecture_type import ArchitectureType

import os
from utils.paths import p


class ModelConfig:
    def __init__(
        self,
        _model_name,
        _data,
        _transformer_config,
        _checkpoint_name=None,
        _num_labels=None,
        _device="cuda:0",
    ):
        # Specific directories
        self.is_downloaded = p.is_dir(os.path.join(p.Config, _model_name))
        self.model_dir = p.get_dir(p.Config, _model_name)
        self.train_dir = p.get_dir(p.Train, _model_name)
        self.data_dir = p.get_dir(p.Data, _data)

        # others
        self.model_name = _model_name
        self.config = _transformer_config
        self.arg_type = self._get_architecture(_model_name)
        self.num_labels = _num_labels
        self.checkpoint_name = _checkpoint_name
        self.device = _device

    @staticmethod
    def _get_architecture(model_name):
        if "bert" in model_name.lower():
            return ArchitectureType.only_encoder
        elif "gpt" in model_name.lower():
            return ArchitectureType.only_decoder
        else:
            return ArchitectureType.both