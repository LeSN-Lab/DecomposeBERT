# In[]: Import Libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# In[]: Define Dataset class
class SDGDataset(Dataset):
    def __init__(self, text_data, labels, tokenizer):
        self.text_data = text_data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=512)
        input_ids = encoding['input_ids']
        return text, input_ids, label


# In[]: Preprocessing
def prep_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    #words = word_tokenize(text)
    words = text
    # Remove stopwords and digits
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    clean_text = ' '.join(words)
    return clean_text


def collate_fn(batch):
    # Separate source and target elements in the batch
    texts, input_ids, labels = zip(*batch)

    # Pad the sequences
    input_ids_tensors = [torch.tensor(ids) for ids in input_ids]
    input_ids_padded = pad_sequence(input_ids_tensors, batch_first=True, padding_value=0)
    labels_tensors = torch.tensor(labels)

    return texts, input_ids_padded, labels_tensors


# In[]: Load preprocessed data

def load_sdg(tokenizer, test_size=0.3, batch_size=128):
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./data/SDG'):
        os.mkdir('./data/SDG')
    if not os.path.isfile('./data/SDG/testDataset.csv'):
        print("Downloading SDG dataset...")
        df = pd.read_csv('https://zenodo.org/record/5550238/files/osdg-community-dataset-v21-09-30.csv?download=1', sep='\t')
        df.to_csv('./data/SDG/testDataset.csv')
    df = pd.read_csv('./data/SDG/testDataset.csv')
    df['clean_text'] = df['text'].apply(lambda x: prep_text(x))

    dataset = SDGDataset(df['clean_text'], df['sdg'], tokenizer)
    train_size = int(len(dataset) * (1 - test_size))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=2021)

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader