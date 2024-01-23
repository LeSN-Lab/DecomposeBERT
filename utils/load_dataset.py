# In[]
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def load_sgd(tokenizer, test_size=0.3, batch_size=128):
    # In[]
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./data/SDG'):
        os.mkdir('./data/SDG')
    if not os.path.isfile('./data/SDG/testDataset.csv'):
        print("Downloading SDG dataset...")
        df = pd.read_csv('https://zenodo.org/record/5550238/files/osdg-community-dataset-v21-09-30.csv?download=1', sep='\t')
        df.to_csv('./data/SDG/testDataset.csv')

    df = pd.read_csv('./data/SDG/testDataset.csv')

    # Tokenization and Padding
    tokenized_texts = tokenizer.batch_encode_plus(
        df['text'].tolist(),
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )

    input_ids = tokenized_texts['input_ids']
    attention_masks = tokenized_texts['attention_mask']

    # Splitting datasets
    X_train, X_test, y_train, y_test = train_test_split(
        input_ids,
        df['sdg'],
        test_size=test_size,
        random_state=42
    )

    train_masks, validation_masks = train_test_split(
        attention_masks,
        test_size=test_size,
        random_state=42
    )

    # Tensor transformation
    train_inputs = torch.tensor(X_train)
    train_labels = torch.tensor(y_train.to_numpy())
    validation_inputs = torch.tensor(X_test)
    validation_labels = torch.tensor(y_test.to_numpy())

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader