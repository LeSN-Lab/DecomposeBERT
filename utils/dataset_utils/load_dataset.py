# In[]: Import Libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.paths import Paths
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataConfig:
    def __init__(self, batch_size=16, test_size=0.3):
        self.data_dir = None
        self.max_length = None
        self.vocab_size = None
        self.vocab_file = None
        self.batch_size = batch_size
        self.test_size = test_size


# In[]: Define load datasets for pretrained
def load_dataloader(data_config, df, tokenizer, test=False, part="Train"):
    prep_texts_path = os.path.join(data_config.prep_dir, f"{part}_texts.pt")
    tokens_path = os.path.join(data_config.prep_dir, f"{part}_tokens.pt")
    labels_path = os.path.join(data_config.prep_dir, f"{part}_labels.pt")

    if Paths.is_file([prep_texts_path, tokens_path, labels_path]):
        print("Loading preprocessed data...")
        prep_texts = torch.load(prep_texts_path)
        tokens_data = torch.load(tokens_path)
        df_y = torch.load(labels_path)
    else:
        print("Preprocessing texts...")
        tqdm.pandas(desc="Preprocessing texts")
        text_list = df[data_config.text_column].to_list()
        prep_texts = preprocess_texts(text_list)
        print("Preprocessing has been done.")

        # Tokenize and encode sequences
        tokens_data = tokenizer.batch_encode_plus(
            prep_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=data_config.max_length,
        )

        # Extract labels
        df_y = df[data_config.label_column].values

        torch.save(prep_texts, prep_texts_path)
        torch.save(tokens_data, tokens_path)
        torch.save(df_y, labels_path)

    # Create the TextDataset
    data = TextDataset(tokens_data, df_y)

    # Define the dataloader
    dataloader = DataLoader(
        data,
        batch_size=data_config.batch_size,
        sampler=SequentialSampler(data) if test else RandomSampler(data),
    )

    return dataloader


# In[]: SDG dataset loader
def load_sdg(tokenizer, data_config):
    dataset = load_dataset("albertmartinez/OSDG")
    train_test_split = dataset["train"].train_test_split(test_size=data_config.test_size, random_state=2024)
    dataset_split = DatasetDict(
        {
            "train": train_test_split["train"],
            "valid": train_test_split["test"],
            "test": dataset["test"],
        }
    )
    data_config.text_column = "text"
    data_config.label_column = "labels"
    return dataset_split["train"], dataset_split["valid"], dataset_split["test"]


# In[]: Yahoo dataset loader
def load_yahoo(tokenizer, data_config):
    dataset = load_dataset("yahoo_answers_topics")
    train_test_split = dataset["train"].train_test_split(test_size=data_config.test_size, random_state=2024)

    data_config.text_column = "question_title"
    data_config.label_column = "topic"
    data_config.max_length = 512

    if not os.path.isfile(file_path):
        print("Downloading Yahoo dataset...")
        try:
            df = pd.read_csv(
                "https://zenodo.org/record/10579179/files/osdg-community-data-v2024-01-01.csv?download=1",
                sep="\t",
            )
            df["sdg"] = df["sdg"] - 1
            df.to_csv(file_path)
            print("Download has been completed")
        except:
            print("Failed to download")

    df = pd.read_csv(file_path)

    train_df, temp_df = train_test_split(
        df, random_state=2018, test_size=data_config.test_size, stratify=df["sdg"]
    )

    valid_df, test_df = train_test_split(
        temp_df, random_state=2018, test_size=0.5, stratify=temp_df["sdg"]
    )

    train_dataloader = load_dataloader(
        data_config=data_config,
        df=train_df,
        tokenizer=tokenizer,
        test=False,
        part="Train",
    )
    valid_dataloader = load_dataloader(
        data_config=data_config,
        df=valid_df,
        tokenizer=tokenizer,
        test=True,
        part="Valid",
    )
    test_dataloader = load_dataloader(
        data_config=data_config, df=test_df, tokenizer=tokenizer, test=True, part="Test"
    )

    return train_dataloader, valid_dataloader, test_dataloader


def load_imdb(data_config):
    dataset = load_dataset("imdb")
    train_test_split = dataset["train"].train_test_split(test_size=data_config.test_size, random_state=2024)
    dataset_split = DatasetDict(
        {
            "train": train_test_split["train"],
            "valid": train_test_split["test"],
            "test": dataset["test"],
        }
    )
    data_config.text_column = "text"
    data_config.label_column = "label"
    return dataset_split["train"], dataset_split["valid"], dataset_split["test"]


def load_data(model_config, batch_size=32, test_size=0.3):
    data_config = DataConfig(batch_size, test_size)
    if model_config.data == "SDG":
        return load_sdg(data_config)
    elif model_config.data == "Yahoo":
        return load_yahoo(data_config)
    elif model_config.data == "IMDB":
        return load_imdb(data_config)
