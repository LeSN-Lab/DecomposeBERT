from utils.model_utils.constants import get_architecture_type
import os
from utils.paths import p


class ModelConfig:
    def __init__(
        self,
        _model_dir,
        _model_name,
        _model_type="Bert",
        _checkpoint_name=None,
        _num_labels=None,
        _device="cuda:0"
    ):
        # Specific directories
        self.is_downloaded = p.is_dir(os.path.join(p.Config, _model_dir))
        self.model_dir = p.get_dir(p.Config, _model_dir)
        self.train_dir = p.get_dir(p.Train, _model_dir)
        self.data_dir = p.get_dir(p.Data, _model_dir)

        # others
        self.model_name = _model_name
        self.model_type = get_architecture_type(_model_type)
        self.num_labels = _num_labels
        self.checkpoint_name = _checkpoint_name
        self.device = _device
