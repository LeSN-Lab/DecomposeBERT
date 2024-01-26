# In[]: Import Libraries
import os
import pandas as pd
import tensorflow_datasets as tfds
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
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[]: Define one-hot-encoding function
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


# In[]: Define load datasets for pretrained
def load_pretrained(df, text_column, label_column, tokenizer, batch_size, _one_hot_encode=True):
    # split train dataset into train, validation and test sets
    train_text, temp_text, train_labels, temp_labels = train_test_split(df[text_column], df[label_column],
                                                                        random_state=2018,
                                                                        test_size=0.3,
                                                                        stratify=df[label_column])

    valid_text, test_text, valid_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=2018,
                                                                    test_size=0.5,
                                                                    stratify=temp_labels)

    # get a number of classified labels
    num_classes = len(np.unique(df[label_column]))

    if _one_hot_encode:
        train_labels = one_hot_encode(train_labels, num_classes)
        valid_labels = one_hot_encode(valid_labels, num_classes)
        test_labels = one_hot_encode(test_labels, num_classes)

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512,
    )
    # tokenize and encode sequences in the validation set
    tokens_valid = tokenizer.batch_encode_plus(
        valid_text.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512,
    )
    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512,
    )

    # define datasets
    train_data = TextDataset(tokens_train, train_labels)
    valid_data = TextDataset(tokens_valid, valid_labels)
    test_data = TextDataset(tokens_test, test_labels)

    # define dataloaders for datasets
    train_dataloader = DataLoader(
        train_data, sampler=RandomSampler(train_data), batch_size=batch_size
    )
    valid_dataloader = DataLoader(
        valid_data, sampler=SequentialSampler(valid_data), batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_data, sampler=SequentialSampler(test_data), batch_size=batch_size
    )

    return train_dataloader, valid_dataloader, test_dataloader


# In[]: Define sdg dataset loader
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
    return load_pretrained(df, 'text', 'sdg', tokenizer, batch_size)


# In[]: Define math dataset loader
def load_math_dataset(hot_encode=True, inputTimestep=20):
    data = \
        tfds.as_numpy(tfds.load('math_qa',
                                batch_size=-1))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    list_classes = {'gain': 0, 'general': 1, 'geometry': 2, 'other': 3,
                    'physics': 4, 'probability': 5}
    for i in range(len(data['test']['category'])):
        y_test.append(list_classes[data['test']['category'][i].decode('utf-8')])
        x_test.append(data['test']['Problem'][i].decode('utf-8'))
    for i in range(len(data['train']['category'])):
        y_train.append(list_classes[data['train']['category'][i].decode('utf-8')])
        x_train.append(data['train']['Problem'][i].decode('utf-8'))

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if hot_encode:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 6

    vocab_size = 5000
    # inputTimestep = 20
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=inputTimestep, padding="pre", truncating="post")

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=inputTimestep, padding="pre", truncating="post")

    return x_train, x_test, y_train, y_test, vocab_size, inputTimestep, num_tags

def load_math_dataset2(hot_encode=True, inputTimestep=20):
    data = \
        tfds.as_numpy(tfds.load('math_qa',
                                batch_size=-1))
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    list_classes = {'gain': 0, 'general': 1, 'geometry': 2, 'other': 3,
                    'physics': 4, 'probability': 5}
    for i in range(len(data['test']['category'])):
        y_test.append(list_classes[data['test']['category'][i].decode('utf-8')])
        x_test.append(data['test']['Problem'][i].decode('utf-8'))
    for i in range(len(data['train']['category'])):
        y_train.append(list_classes[data['train']['category'][i].decode('utf-8')])
        x_train.append(data['train']['Problem'][i].decode('utf-8'))

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if hot_encode:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        num_tags = y_train.shape[1]
    else:
        num_tags = 6

    vocab_size = 5000
    # inputTimestep = 20
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=inputTimestep, padding="pre", truncating="post")

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=inputTimestep, padding="pre", truncating="post")

    return x_train, x_test, y_train, y_test, vocab_size, inputTimestep, num_tags

