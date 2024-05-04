# In[]: Import Libraries
import os
from os.path import join as join
from torch.utils.data import DataLoader
from utils.paths import p, get_dir
from datasets import load_dataset, DatasetDict
import torch


class DataConfig:
    data_dir = None
    max_length = None
    vocab_size = None
    batch_size = None
    test_size = None
    seed = None


data_config = DataConfig()
# In[]: Define load datasets for pretrained
def load_dataloader(model_config, dataset=None, skip=False):
    if dataset is not None:
        get_dir(model_config.data_dir, True)
        shuffle_train = dataset["train"].shuffle(seed=data_config.seed)
        train_test_split = shuffle_train.train_test_split(test_size=data_config.test_size)
        dataset = DatasetDict(
            {
                "train": train_test_split["train"],
                "valid": train_test_split["test"],
                "test": dataset["test"],
            }
        )
        torch.save(dataset, join(model_config.data_dir, "dataset.pt"))

    else:
        dataset = torch.load(join(model_config.data_dir, "dataset.pt"))

    train_data = dataset["train"]
    valid_data = dataset["valid"]
    test_data = dataset["test"]

    # Define the dataloader
    train_dataloader = DataLoader(train_data, batch_size=data_config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=data_config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=data_config.batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


# In[]: SDG dataset loader
def load_sdg(model_config):
    data_config.data_dir = get_dir(join(p.Data, "OSDG"))
    data_config.text_column = "text"
    data_config.label_column = "labels"
    if get_dir(data_config.data_dir):
        return load_dataloader(model_config)
    else:
        dataset = load_dataset("albertmartinez/OSDG")
        return load_dataloader(model_config, dataset)


# In[]: Yahoo dataset loader
def load_yahoo(model_config):
    data_config.data_dir = get_dir(join(p.Data, "Yahoo"))
    data_config.text_column = "question_title"
    data_config.label_column = "topic"
    if get_dir(data_config.data_dir):
        return load_dataloader(model_config)
    else:
        dataset = load_dataset("yahoo_answers_topics")
        return load_dataloader(model_config, dataset)


def load_imdb(model_config):
    data_config.data_dir = get_dir(join(p.Data, "IMDb"))
    data_config.text_column = "text"
    data_config.label_column = "label"
    if get_dir(data_config.data_dir):
        return load_dataloader(model_config)
    else:
        dataset = load_dataset("imdb")
        return load_dataloader(model_config, dataset)


def load_data(model_config, batch_size=32, test_size=0.3, seed=42):
    data_config.batch_size = batch_size
    data_config.test_size = test_size
    data_config.seed = seed
    if model_config.data == "OSDG":
        return load_sdg(model_config)
    elif model_config.data == "Yahoo":
        return load_yahoo(model_config)
    elif model_config.data == "IMDb":
        return load_imdb(model_config)
