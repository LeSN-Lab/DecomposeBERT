# In[]: Import Libraries
import os
from os.path import join as join
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
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

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        data = self.tokenized_data[index]
        return {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
            'labels': data['label']
        }

# In[]: Define load datasets for pretrained
def load_dataloader(model_config, dataset=None, skip=False):
    if dataset is not None:
        get_dir(model_config.data_dir, True)
        shuffle_train = dataset["train"].shuffle(seed=data_config.seed)
        train_test_split = shuffle_train.train_test_split(
            test_size=data_config.test_size
        )
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

    return train_data, valid_data, test_data


# In[]: SDG dataset loader
def load_sdg(model_config):
    data_config.data_dir = get_dir(join(p.Data, model_config.data), True)
    data_config.text_column = "text"
    data_config.label_column = "labels"
    if get_dir(join(data_config.data_dir, "dataset.pt")):
        return load_dataloader(model_config)
    else:
        dataset = load_dataset("albertmartinez/OSDG")
        return load_dataloader(model_config, dataset)


# In[]: Yahoo dataset loader
def load_yahoo(model_config):
    data_config.data_dir = get_dir(join(p.Data, model_config.data))
    data_config.text_column = "question_title"
    data_config.label_column = "topic"
    if get_dir(join(data_config.data_dir, "dataset.pt")):
        return load_dataloader(model_config)
    else:
        dataset = load_dataset("yahoo_answers_topics")
        return load_dataloader(model_config, dataset)


def load_imdb(model_config):
    data_config.data_dir = get_dir(join(p.Data, model_config.data))
    data_config.text_column = "text"
    data_config.label_column = "label"
    if get_dir(join(data_config.data_dir, "dataset.pt")):
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


data_config = DataConfig()
