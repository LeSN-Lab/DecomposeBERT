# In[]: Import Libraries
import os
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils.dataset_utils.text_preprocessing import preprocess_texts
from utils.paths import Paths
from tqdm.auto import tqdm
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[]: Define Dataset class
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class DataConfig:
    def __init__(self, batch_size=16, test_size=0.3):
        self.text_column = None
        self.label_column = None
        self.data_dir = None
        self.prep_dir = None
        self.batch_size = batch_size
        self.test_size = test_size
        self.max_length = None


# In[]: Define load datasets for pretrained
def load_dataloader(
    data_config, df, tokenizer, test=False, part="Train"
):
    prep_texts_path = os.path.join(data_config.prep_dir, f'{part}_texts.pt')
    tokens_path = os.path.join(data_config.prep_dir, f'{part}_tokens.pt')
    labels_path = os.path.join(data_config.prep_dir, f'{part}_labels.pt')

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
    print("Loading dataset")

    file_path = os.path.join(data_config.data_dir, "Dataset.csv")
    data_config.text_column = "text"
    data_config.label_column = "sdg"
    data_config.max_length = 512

    if not os.path.isfile(file_path):
        print("Downloading SDG dataset...")
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
        data_config=data_config, df=train_df, tokenizer=tokenizer, test=False, part="Train"
    )
    valid_dataloader = load_dataloader(
        data_config=data_config, df=valid_df, tokenizer=tokenizer, test=True, part="Valid"
    )
    test_dataloader = load_dataloader(
        data_config=data_config, df=test_df, tokenizer=tokenizer, test=True, part="Test"
    )

    return train_dataloader, valid_dataloader, test_dataloader


# In[]: Math dataset loader
def load_math_dataset(tokenizer, batch_size=32, max_length=20):
    # Load the dataset
    data, info = tfds.load(
        "math_qa", split=["train", "test", "validation"], batch_size=-1, with_info=True
    )
    train_data, test_data, validation_data = data

    # Initialize lists
    x_train, y_train = [], []
    x_valid, y_valid = [], []
    x_test, y_test = [], []

    # Define classes
    list_classes = {
        "gain": 0,
        "general": 1,
        "geometry": 2,
        "other": 3,
        "physics": 4,
        "probability": 5,
    }

    # Process test data
    for i in range(len(test_data["category"])):
        category = test_data["category"][i].numpy().decode("utf-8")
        y_test.append(list_classes[category])
        x_test.append(test_data["Problem"][i].numpy().decode("utf-8"))

    # Process train data
    for i in range(len(train_data["category"])):
        category = train_data["category"][i].numpy().decode("utf-8")
        y_train.append(list_classes[category])
        x_train.append(train_data["Problem"][i].numpy().decode("utf-8"))

    # Process valid data
    for i in range(len(validation_data["category"])):
        category = validation_data["category"][i].numpy().decode("utf-8")
        y_valid.append(list_classes[category])
        x_valid.append(validation_data["Problem"][i].numpy().decode("utf-8"))

    # Convert to pandas DataFrame
    train_df = pd.DataFrame({"Problem": x_train, "Category": y_train})
    test_df = pd.DataFrame({"Problem": x_test, "Category": y_test})
    valid_df = pd.DataFrame({"Problem": x_valid, "Category": y_valid})

    # Create dataloader
    train_dataloader = load_dataloader(
        train_df, "Problem", "Category", tokenizer, batch_size, max_length
    )
    valid_dataloader = load_dataloader(
        valid_df, "Problem", "Category", tokenizer, batch_size, max_length
    )
    test_dataloader = load_dataloader(
        test_df, "Problem", "Category", tokenizer, batch_size, max_length
    )

    return train_dataloader, valid_dataloader, test_dataloader


def load_dataset(model_config, tokenizer, batch_size=32, test_size=0.3):
    data_config = DataConfig(batch_size, test_size)
    data_config.data_dir = model_config.data_dir
    data_config.prep_dir = model_config.prep_dir
    if model_config.data == "SDG":
        return load_sdg(tokenizer, data_config)
