import os
import torch
import json
from os.path import join as join
from os.path import exists
from os import makedirs as makedirs
from colorama import Fore, Style, init


class Paths:
    """Class for managing directory paths."""

    def __init__(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.root = os.path.dirname(current_path)
        os.chdir(self.root)

        # Set roots
        self.Datasets = Paths.get_dir("Datasets")
        self.Models = Paths.get_dir("Models")
        self.Modules = Paths.get_dir("Modules")
        self.Checkpoints = Paths.get_dir("Checkpoint")

    @staticmethod
    def get_dir(path):
        makedirs(path, exist_ok=True)
        return path


class ModelConfig:
    def __init__(
        self,
        model_name,
        device: torch.device = torch.device("cuda:0"),
    ):
        """
        Initialize the configuration for the model.

        Args:
            model_name (str): The name of the model.
            device (torch.device): The device to run the model on. Defaults to "cuda:0".
        """
        # Specific directories

        self.config = ModelConfig.load_config(model_name)
        self.model_name = self.config["model_name"]
        self.task_type = self.config["task_type"]
        self.architectures = self.config["architectures"]
        self.dataset_name = self.config["dataset_name"]
        self.num_labels = self.config["num_labels"]
        self.cache_dir = self.config["cache_dir"]
        self.device = device

        self.Datasets = Paths.get_dir("Datasets")
        self.Models = Paths.get_dir("Models")
        self.Modules = Paths.get_dir("Modules")
        self.Checkpoints = Paths.get_dir("Checkpoint")

    @staticmethod
    def load_config(model_name):
        with open("utils/config.json", "r") as json_file:
            config = json.load(json_file)
            model_config = config["model"][model_name]
        return model_config

    def summary(self):
        color_print(self.config)


class DataConfig:
    def __init__(
        self,
        dataset_name,
        device: torch.device = torch.device("cuda:0"),
        max_length=512,
        batch_size=4,
        valid_size=0.1,
        seed=42,
        return_fields=["input_ids", "attention_mask", "labels"],
        do_cache=True,
    ):
        self.config = DataConfig.load_config(dataset_name)
        self.device = device
        self.dataset_name = self.config["dataset_name"]
        self.cache_dir = self.config["cache_dir"]
        self.text_column = self.config["text_column"]
        self.label_column = self.config["label_column"]
        self.task_type = self.config["task_type"]
        if self.config["path"] == "code_search_net":
            self.dataset_args = {
                "path": self.config["path"],
                "config_name": self.config["config_name"],
            }
        else:
            self.dataset_args = {"path": self.config["path"]}
        self.max_length = max_length
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.seed = seed
        self.return_fields = return_fields

        self.do_cache = do_cache

    def is_cached(self):
        train = join(self.cache_dir, "train.pt")
        valid = join(self.cache_dir, "valid.pt")
        test = join(self.cache_dir, "test.pt")
        return exists(train) and exists(valid) and exists(test)

    @staticmethod
    def load_config(dataset_name):
        with open("utils/config.json", "r") as json_file:
            config = json.load(json_file)
            data_config = config["dataset"][dataset_name]
        return data_config

    def summary(self):
        color_print(self.config)


def safe_std(tensor, epsilon=None, unbiased=False, dim=None, keepdim=True):
    if tensor.numel():
        return nanstd(tensor, dim=dim, unbiased=unbiased, keepdim=keepdim)
    else:
        return torch.tensor(epsilon, dtype=tensor.dtype)


def nanstd(tensor, unbiased=False, dim=None, keepdim=True):
    mask = torch.isnan(tensor)
    n_obs = mask.logical_not().sum(dim=dim, keepdim=keepdim)
    mean = torch.nanmean(tensor, dim=dim, keepdim=keepdim)

    centered = tensor - mean
    centered = centered.masked_fill(mask, 0)
    sum_sq = torch.sum(centered**2, dim=dim, keepdim=keepdim)

    unbiased_factor = torch.where(n_obs > 1, n_obs - 1, n_obs)
    var = sum_sq / unbiased_factor

    std = torch.sqrt(var)
    if not keepdim:
        std = std.squeeze(dim)

    return std


def color_print(data):
    init(autoreset=True)
    print(f"{Fore.CYAN}{data}{Style.RESET_ALL}")


def set_parameters(layer, weight, bias):
    layer.weight = torch.nn.Parameter(weight)
    layer.bias = torch.nn.Parameter(bias)

