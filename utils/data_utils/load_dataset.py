# In[]: Import Libraries
import os
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from text_preprocessing import pre_processing


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


# In[]: Define load datasets for pretrained
def load_dataloader(df, text_column, label_column, tokenizer, batch_size, max_length, test=False):
    df_x = df[text_column].apply(lambda x: pre_processing(x))
    df_y = df[label_column].values
    # Tokenize and encode sequences
    tokens_df = tokenizer.batch_encode_plus(
        df_x.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=max_length,
    )

    # Create the TextDataset
    data = TextDataset(tokens_df, df_y)

    # Define the dataloader
    if test:
        dataloader = DataLoader(
            data, sampler=SequentialSampler(data), batch_size=batch_size
        )
    else:
        dataloader = DataLoader(
            data, sampler=RandomSampler(data), batch_size=batch_size
        )

    return dataloader


# In[]: SDG dataset loader
def load_sdg(tokenizer=None, batch_size=32):
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    if not os.path.isdir("./data/SDG"):
        os.mkdir("./data/SDG")

    if not os.path.isfile("./data/SDG/Dataset.csv"):
        print("Downloading SDG dataset...")
        df = pd.read_csv('https://zenodo.org/record/10579179/files/osdg-community-data-v2024-01-01.csv?download=1', sep='\t')
        df.to_csv('./data/SDG/Dataset.csv')
        print("Download has been completed")

    df = pd.read_csv('./data/SDG/Dataset.csv')
    df['sdg'] = df['sdg'] - 1
    train_df, temp_df = train_test_split(df, random_state=2018, test_size=0.3, stratify=df["sdg"])

    valid_df, test_df = train_test_split(temp_df, random_state=2018, test_size=0.5, stratify=temp_df["sdg"])
    if tokenizer is not None:
        train_dataloader = load_dataloader(train_df, 'text', 'sdg', tokenizer, batch_size, 512)
        valid_dataloader = load_dataloader(valid_df, 'text', 'sdg', tokenizer, batch_size, 512)
        test_dataloader = load_dataloader(test_df, 'text', 'sdg', tokenizer, batch_size, 512)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    return train_dataloader, valid_dataloader, test_dataloader


# In[]: Math dataset loader
def load_math_dataset(tokenizer, batch_size=32, max_length=20):
    # Load the dataset
    data, info = tfds.load('math_qa', split=['train', 'test', 'validation'], batch_size=-1, with_info=True)
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
