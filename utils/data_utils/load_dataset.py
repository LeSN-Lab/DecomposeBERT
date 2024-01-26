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
def load_dataloader(df, text_column, label_column, tokenizer, batch_size, _one_hot_encode=True, test=False):
    df_x = df[text_column]
    df_y = df[label_column]

    # get a number of classified labels
    num_classes = len(np.unique(df_y))

    if _one_hot_encode:
        df_y = one_hot_encode(df_y, num_classes)

    # tokenize and encode sequences in the training set
    tokens_df = tokenizer.batch_encode_plus(
        df_x.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512,
    )

    # define datasets
    data = TextDataset(tokens_df, df_y)

    # define dataloaders for datasets
    if test:
        dataloader = DataLoader(
            data, sampler=SequentialSampler(data), batch_size=batch_size
        )
    else:
        dataloader = DataLoader(
            data, sampler=RandomSampler(data), batch_size=batch_size
        )

    return dataloader


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
    train_df, temp_df = train_test_split(df,
                                         random_state=2018,
                                         test_size=0.3,
                                         stratify=df["sdg"])

    valid_df, test_df = train_test_split(temp_df,
                                         random_state=2018,
                                         test_size=0.5,
                                         stratify=temp_df["sdg"])
    train_dataloader = load_dataloader(df, 'text', 'sdg', tokenizer, batch_size)
    valid_dataloader = load_dataloader(df, 'text', 'sdg', tokenizer, batch_size)
    test_dataloader = load_dataloader(df, 'text', 'sdg', tokenizer, batch_size)

    return train_dataloader, valid_dataloader, test_dataloader


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

    train_df = pd.DataFrame({'Problem': x_train, 'Category': y_train})
    test_df = pd.DataFrame({'Problem': x_test, 'Category': y_test})
    return train_df, test_df

#
#     if hot_encode:
#         y_train = to_categorical(y_train)
#         y_test = to_categorical(y_test)
#         num_tags = y_train.shape[1]
#     else:
#         num_tags = 6
#
#     vocab_size = 5000
#     # inputTimestep = 20
#     oov_tok = "<OOV>"
#
#     tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
#     tokenizer.fit_on_texts(x_train)
#     tokenizer.fit_on_texts(x_test)
#
#     x_train = tokenizer.texts_to_sequences(x_train)
#     x_train = pad_sequences(x_train, maxlen=inputTimestep, padding="pre", truncating="post")
#
#     x_test = tokenizer.texts_to_sequences(x_test)
#     x_test = pad_sequences(x_test, maxlen=inputTimestep, padding="pre", truncating="post")
#
#     return x_train, x_test, y_train, y_test, vocab_size, inputTimestep, num_tags

df_t, df_T = load_math_dataset()

print(len(df_t))
print(len(df_T))