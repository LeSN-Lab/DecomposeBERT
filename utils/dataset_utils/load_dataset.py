# In[]: Import Libraries
import os
from os.path import join
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import shuffle
from datasets import load_dataset, DatasetDict
from utils.paths import paths, get_dir
from utils.model_utils.load_model import load_tokenizer


class DataConfig:
    def __init__(
        self,
        cached_dir,
        max_length=512,
        vocab_size=None,
        batch_size=4,
        valid_size=0.1,
        seed=42,
        return_fields=["input_ids", "attention_mask", "labels"],
        do_cache=True
    ):
        self.dataset_name = None
        self.cached_dir = cached_dir
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.seed = seed
        self.return_fields = return_fields
        self.text_column = "text"
        self.label_column = "labels"
        self.task_type = "classification"
        self.do_cache = do_cache

    def is_cached(self):
        train = join(self.cached_dir, "train.pt")
        valid = join(self.cached_dir, "valid.pt")
        test = join(self.cached_dir, "test.pt")
        return get_dir(train) and get_dir(valid) and get_dir(test)


class CustomDataset(Dataset):
    def __init__(self, data, data_config):
        self.data = data
        self.data_config = data_config

    def __len__(self):
        return len(self.data[self.data_config.return_fields[0]])

    def __getitem__(self, index):
        return {key: self.data[key][index] for key in self.data_config.return_fields}


def tokenize_dataset(raw_dataset, tokenizer, data_config):
    tokenized_datasets = {field: [] for field in data_config.return_fields}
    for example in raw_dataset:
        if data_config.task_type == "seq2seq":
            inputs = tokenizer(
                example[data_config.text_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            targets = tokenizer(
                example[data_config.label_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(inputs["input_ids"][0])
            tokenized_datasets["attention_mask"].append(inputs["attention_mask"][0])
            tokenized_datasets["labels"].append(targets["input_ids"][0])
        else:
            tokens = tokenizer(
                example[data_config.text_column],
                padding="max_length",
                truncation=True,
                max_length=data_config.max_length,
                return_tensors="pt",
            )
            tokenized_datasets["input_ids"].append(tokens["input_ids"][0])
            tokenized_datasets["attention_mask"].append(tokens["attention_mask"][0])
            for field in data_config.return_fields:
                if field not in ["input_ids", "attention_mask"]:
                    tokenized_datasets[field].append(example.get(field, 0))  # Default to 0 if not found
    return CustomDataset(tokenized_datasets, data_config)


# In[]: Define load datasets for pretrained
def load_dataloader(dataset, tokenizer, data_config, is_valid=False):
    if is_valid:
        shuffled_dataset = dataset.shuffle(seed=data_config.seed)
        tokenized_dataset = tokenize_dataset(shuffled_dataset, tokenizer, data_config)
        train_valid_split = tokenized_dataset.train_test_split(test_size=data_config.valid_size)
        train_dataset = train_valid_split['train']
        valid_dataset = train_valid_split['test']

        train_dataloader = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=data_config.batch_size, shuffle=False)
        return train_dataloader, valid_dataloader
    else:
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, data_config)
        dataloader = DataLoader(tokenized_dataset, batch_size=data_config.batch_size, shuffle=False)
        return dataloader

def load_cached_dataset(data_config, model_config):
    tokenizer = load_tokenizer(model_config)
    cached_dataset_path = data_config.cached_dir

    if not data_config.is_cached() or not data_config.do_cache:  # If not cached, generate caches
        dataset = load_dataset(dataset_map[data_config.dataset_name])
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        train_dataloader, valid_dataloader = load_dataloader(train_dataset, tokenizer, data_config, True)
        test_dataloader = load_dataloader(test_dataset, tokenizer, data_config)

        torch.save(train_dataloader, join(cached_dataset_path, "train.pt"))
        torch.save(valid_dataloader, join(cached_dataset_path, "valid.pt"))
        torch.save(test_dataloader, join(cached_dataset_path, "test.pt"))
        print("Caching is completed.")
    else:
        print("Load cached dataset.")
        train_dataloader = torch.load(join(cached_dataset_path, "train.pt"))
        valid_dataloader = torch.load(join(cached_dataset_path, "valid.pt"))
        test_dataloader = torch.load(join(cached_dataset_path, "test.pt"))
    return train_dataloader, valid_dataloader, test_dataloader


dataset_map = {
    "OSDG": "albertmartinez/OSDG",
    "Yahoo": "yahoo_answers_topics",
    "IMDB": "imdb",
    "Code": "code_search_net",
}


# In[]: SDG dataset loader
def load_sdg(data_config):
    data_config.text_column = "text"
    data_config.label_column = "labels"


# In[]: Yahoo dataset loader
def load_yahoo(data_config):
    data_config.text_column = "question_title"
    data_config.label_column = "topic"


def load_imdb(data_config):
    data_config.text_column = "text"
    data_config.label_column = "label"


def load_code_search_net(data_config):
    data_config.return_fields = ["input_ids", "attention_mask", "labels"]
    data_config.text_column = "func_code_string"
    data_config.label_column = "func_documentation_string"


def load_data(model_config, batch_size=32, valid_size=0.1, seed=42, do_cache=True):
    data_config = DataConfig(
        cached_dir=model_config.data_dir,
        max_length=512,
        vocab_size=None,
        batch_size=batch_size,
        valid_size=valid_size,
        seed=seed,
        return_fields=["input_ids", "attention_mask", "labels"],
        do_cache=do_cache
    )
    data_config.dataset_name = model_config.dataset_name

    if model_config.dataset_name == "OSDG":
        load_sdg(data_config)
    elif model_config.dataset_name == "Yahoo":
        load_yahoo(data_config)
    elif model_config.dataset_name == "IMDB":
        load_imdb(data_config)
    elif model_config.dataset_name == "Code":
        data_config.task_type = model_config.task_type
        load_code_search_net(data_config)
    else:
        raise ValueError(f"Unsupported dataset: {model_config.dataset_name}")
    return load_cached_dataset(data_config, model_config)

def convert_dataset_labels_to_binary(dataloader, target_class, is_stratified=False):
    input_ids, attention_masks, labels = [], [], []
    for batch in dataloader:
        input_ids.append(batch["input_ids"])
        attention_masks.append(batch["attention_mask"])

        binary_labels = (batch["labels"] == target_class).long()
        labels.append(binary_labels)

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    labels = torch.cat(labels)

    if is_stratified:
        # Count the number of samples for each class
        class_0_indices = [i for i, label in enumerate(labels) if label == 0]
        class_1_indices = [i for i, label in enumerate(labels) if label == 1]

        # Find the minimum class size
        min_class_size = min(len(class_0_indices), len(class_1_indices))

        # Convert to tensors and shuffle
        class_0_indices = torch.tensor(class_0_indices)
        class_1_indices = torch.tensor(class_1_indices)

        class_0_indices = class_0_indices[
            torch.randperm(len(class_0_indices))[:min_class_size]
        ]
        class_1_indices = class_1_indices[
            torch.randperm(len(class_1_indices))[:min_class_size]
        ]

        # Combine indices and shuffle them
        balanced_indices = torch.cat([class_0_indices, class_1_indices]).long()
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

        # Subset the data to the balanced indices
        input_ids = input_ids[balanced_indices]
        attention_masks = attention_masks[balanced_indices]
        labels = labels[balanced_indices]

    transformed_dataset = CustomDataset(input_ids, attention_masks, labels)
    transformed_dataloader = DataLoader(
        transformed_dataset, batch_size=dataloader.batch_size, shuffle=not is_stratified
    )

    return transformed_dataloader


def extract_and_convert_dataloader(dataloader, true_index, false_index):
    # Extract the data using the provided indices

    input_ids, attention_masks, labels = [], [], []

    for batch in dataloader:
        mask = (batch["labels"] == true_index) | (batch["labels"] == false_index)
        if mask.any():
            input_ids.append(batch["input_ids"][mask])
            attention_masks.append(batch["attention_mask"][mask])
            labels.append(batch["labels"][mask])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.cat(labels, dim=0)

    subset_dataset = CustomDataset(input_ids, attention_masks, labels)
    subset_dataloader = DataLoader(subset_dataset, batch_size=dataloader.batch_size)

    # Apply convert_dataset_labels_to_binary
    binary_dataloader = convert_dataset_labels_to_binary(subset_dataloader, true_index)

    return binary_dataloader