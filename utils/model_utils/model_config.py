import os
from os.path import join as join
from utils.paths import p, get_dir


class ModelConfig:
    def __init__(
        self,
        _model_name,
        _model_type,
        _data,
        _is_pretrained,
        _transformer_config,
        _checkpoint_name=None,
        _device="cuda:0",
    ):
        # Specific directories
        norm_name = os.path.normpath(_model_name)
        temp = os.path.join(_model_type,norm_name)
        self.is_downloaded = get_dir(join(p.Configs, temp))
        self.data_dir = get_dir(join(p.Data, _data), True)
        self.config_dir = get_dir(join(p.Configs, temp), True)
        self.train_dir = get_dir(join(p.Train, temp), True)
        self.module_dir = get_dir(join(p.Modules, temp), True)
        self.prep_dir = get_dir(self.data_dir, "Prep_data")
        self.module_dir = get_dir(p.Module, temp)

        # others
        self.model_name = norm_name
        self.is_pretrained = _is_pretrained
        self.transformer_config = _transformer_config
        self.checkpoint_name = _checkpoint_name
        self.device = _device

