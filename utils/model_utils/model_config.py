import os
import torch
from os.path import join as join
from utils.paths import paths, get_dir


class ModelConfig:
    def __init__(
        self,
        model_name: str,
        task_type: str,
        dataset_name: str,
        checkpoint: str = None,
        device: torch.device = torch.device("cuda:0"),
    ):
        """
        Initialize the configuration for the model.

        Args:
            model_name (str): The name of the model.
            task_type (str): The task type of the model.
            dataset_name (str): The name of the dataset.
            checkpoint (str, optional): The checkpoint for the model. Defaults to None.
            device (torch.device): The device to run the model on. Defaults to "cuda:0".
        """
        # Specific directories

        self.model_name = model_name
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.checkpoint = checkpoint
        self.device = device

        norm_name = os.path.normpath(model_name)
        temp = os.path.join(task_type, norm_name)
        self.is_downloaded = get_dir(join(paths.Configs, temp))
        self.config_dir = get_dir(join(paths.Configs, temp), True)
        self.module_dir = get_dir(join(paths.Modules, temp), True)
        self.data_dir = get_dir(join(paths.Data, self.dataset_name), True)
