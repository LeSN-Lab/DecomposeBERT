# In[]: Import Libraries
import os
import pandas as pd
import tensorflow_datasets as tfds
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import partial


# In[]: Define Dataset class
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# In[]: Define load datasets for pretrained
def load_dataloader(df, text_column, label_column, tokenizer, batch_size, max_length, test=False):
    df_x = df[text_column]
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
def load_sdg(tokenizer, batch_size=32):
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./data/SDG'):
        os.mkdir('./data/SDG')

    if not os.path.isfile('./data/SDG/Dataset.csv'):
        print("Downloading SDG dataset...")
        df = pd.read_csv('https://zenodo.org/record/5550238/files/osdg-community-dataset-v21-09-30.csv?download=1', sep='\t')
        df.to_csv('./data/SDG/Dataset.csv')
        print("Download has been completed")

    df = pd.read_csv('./data/SDG/Dataset.csv')
    df['sdg'] = df['sdg'] - 1
    train_df, temp_df = train_test_split(df, random_state=2018, test_size=0.3, stratify=df["sdg"])

    valid_df, test_df = train_test_split(temp_df, random_state=2018, test_size=0.5, stratify=temp_df["sdg"])

    train_dataloader = load_dataloader(train_df, 'text', 'sdg', tokenizer, batch_size, 512)
    valid_dataloader = load_dataloader(valid_df, 'text', 'sdg', tokenizer, batch_size, 512)
    test_dataloader = load_dataloader(test_df, 'text', 'sdg', tokenizer, batch_size, 512)

    return train_dataloader, valid_dataloader, test_dataloader


# In[]: Math dataset loader
def load_math_dataset(tokenizer, batch_size=32, max_length=20):
    # Load the dataset
    data = tfds.as_numpy(tfds.load('math_qa', batch_size=-1))

    # Initialize lists
    x_train, y_train, x_test, y_test = [], [], [], []

    # Define classes
    list_classes = {'gain': 0, 'general': 1, 'geometry': 2, 'other': 3, 'physics': 4, 'probability': 5}

    # Process test data
    for i in range(len(data['test']['category'])):
        y_test.append(list_classes[data['test']['category'][i].decode('utf-8')])
        x_test.append(data['test']['Problem'][i].decode('utf-8'))

    # Process train data
    for i in range(len(data['train']['category'])):
        y_train.append(list_classes[data['train']['category'][i].decode('utf-8')])
        x_train.append(data['train']['Problem'][i].decode('utf-8'))

    # Convert to numpy arrays
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Create pandas dataFrames
    train_df = pd.DataFrame({'Problem': x_train, 'Category': y_train})
    test_df = pd.DataFrame({'Problem': x_test, 'Category': y_test})

    # Splint train data into train and validation data
    train_df, valid_df = train_test_split(train_df, random_state=2018, test_size=0.1, stratify=train_df['Category'])

    train_dataloader = load_dataloader(train_df, 'Problem', 'Category', tokenizer, batch_size, max_length)
    valid_dataloader = load_dataloader(valid_df, 'Problem', 'Category', tokenizer, batch_size, max_length)
    test_dataloader = load_dataloader(test_df, 'Problem', 'Category', tokenizer, batch_size, max_length)

    return train_dataloader, valid_dataloader, test_dataloader

