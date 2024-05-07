# In[]: Import Libraries
import os
from os.path import join as join
from torch.utils.data import DataLoader
from utils.model_utils.load_tokenizer import load_tokenizer
from utils.paths import p, get_dir
from datasets import load_dataset, DatasetDict
import torch


class DataConfig:
    def __init__(self):
        self.data_dir = None
        self.max_length = 512
        self.vocab_size = None
        self.batch_size = None
        self.test_size = None
        self.seed = None
        self.text_column = None
        self.label_column = None


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


# In[]: Define load datasets for pretrained
def load_dataloader(model_config, dataset=None):
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

    tokenizer = load_tokenizer(model_config)
    tokenized_datasets = {}

    for split, data in dataset.items():
        texts = [example[data_config.text_column] for example in data]
        tokenized_batch = tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=data_config.max_length,
        )
        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]

        labels = data[data_config.label_column]

        tokenized_datasets[split] = TextDataset(input_ids, attention_mask, labels)

    train_dataloader = DataLoader(
        tokenized_datasets.get("train"), batch_size=data_config.batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        tokenized_datasets.get("valid"), batch_size=data_config.batch_size
    )
    test_dataloader = DataLoader(
        tokenized_datasets.get("test"), batch_size=data_config.batch_size
    )

    return train_dataloader, valid_dataloader, test_dataloader


def convert_dataset_labels_to_binary(dataloader, target_class):
    input_ids, attention_masks, labels = [], [], []
    for batch in dataloader:
        input_ids.append(batch["input_ids"])
        attention_masks.append(batch["attention_mask"])

        binary_labels = (batch["labels"] == target_class).long()
        labels.append(binary_labels)

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    labels = torch.cat(labels)

    transformed_dataset = TextDataset(input_ids, attention_masks, labels)
    transformed_dataloader = DataLoader(
        transformed_dataset,
        batch_size=dataloader.batch_size,
    )

    return transformed_dataloader


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
    data_config.data_dir = get_dir(join(p.Data, model_config.data), True)
    data_config.text_column = "question_title"
    data_config.label_column = "topic"
    if get_dir(join(data_config.data_dir, "dataset.pt")):
        return load_dataloader(model_config)
    else:
        dataset = load_dataset("yahoo_answers_topics")
        return load_dataloader(model_config, dataset)


def load_imdb(model_config):
    data_config.data_dir = get_dir(join(p.Data, model_config.data), True)
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
