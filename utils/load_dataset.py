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
import multiprocessing
from functools import partial

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
        attention_mask = encoding['attention_mask']
        return text, input_ids, attention_mask, label


# In[]: Preprocessing
def prep_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    words = word_tokenize(text)
    words = [word for word in words if word.isalpha() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def parallel_prep_texts(texts, stop_words, n_jobs=4):
    lemmatizer = WordNetLemmatizer()
    with multiprocessing.Pool(n_jobs) as pool:
        clean_texts = pool.map(partial(prep_text, lemmatizer=lemmatizer, stop_words=stop_words), texts)
    return clean_texts


def collate_fn(batch):
    # Separate source and target elements in the batch
    texts, input_ids, attention_masks, labels = zip(*batch)

    input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
    labels_tensors = torch.tensor(labels, dtype=torch.long)

    if not input_ids_tensors:  # Check if all sequences are empty
        return None

    attention_masks_tensors = [torch.tensor(mask, dtype=torch.long) for mask in attention_masks if isinstance(mask, (list, torch.Tensor)) and len(mask) > 0]

    input_ids_padded = pad_sequence(input_ids_tensors, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks_tensors, batch_first=True, padding_value=0)

    return texts, input_ids_padded, attention_masks_padded, labels_tensors


# In[]: Load preprocessed data

def load_sdg(tokenizer, test_size=0.3, val_size=0.1, batch_size=128):
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./data/SDG'):
        os.mkdir('./data/SDG')

    if not os.path.isfile('./data/SDG/Dataset.csv'):
        print("Downloading SDG dataset...")
        df = pd.read_csv('https://zenodo.org/record/5550238/files/osdg-community-dataset-v21-09-30.csv?download=1', sep='\t')
        df.to_csv('./data/SDG/Dataset.csv')
        print("Download has been completed")

    if not os.path.isfile('./data/SDG/prepData.csv'):
        df = pd.read_csv('./data/SDG/Dataset.csv')
        stop_words = set(stopwords.words('english'))
        print("Preprocessing texts...")
        df['clean_text'] = parallel_prep_texts(df['text'].tolist(), stop_words, n_jobs=4)
        df.to_csv('./data/SDG/prepData.csv', index=False)
        print("Processing texts has been completed")

    df = pd.read_csv('./data/SDG/prepData.csv')

    dataset = SDGDataset(df['clean_text'], df['sdg'], tokenizer)
    train_size = int(len(dataset) * (1 - test_size))
    test_size = len(dataset) - train_size
    print(f"train size: {train_size} test size: {test_size}")
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=2021)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=val_size,
                                                  random_state=2021)
    trainDataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, collate_fn=collate_fn
    )
    valDataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size, collate_fn=collate_fn
    )
    testDataloader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, collate_fn=collate_fn
    )

    return trainDataloader, valDataloader, testDataloader
