# In[]: Import libraries
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler





# In[]: Preprocessing

def token_embeddings():
    '''A [CLS] token is added to the input word tokens at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.'''
    pass

def segment_embeddings():
    '''A marker indicating Sentence A or Sentence B is added to each token. This allows the encoder to distinguish between sentences.'''
    pass

def positional_embeddings():
    '''A positional embedding is added to each token to indicate its position in the sentence.'''
pass


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


def text_processing(tweet, decode=True):
    # remove https links
    if decode:
        tweet = tweet.decode('utf-8')
    clean_tweet = re.sub(r'http\S+', '', tweet)
    # remove punctuation marks
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    clean_tweet = ''.join(ch for ch in clean_tweet if ch not in set(punctuation))
    # convert text to lowercase
    clean_tweet = clean_tweet.lower()
    # remove numbers
    clean_tweet = re.sub('\d', ' ', clean_tweet)
    # remove whitespaces
    clean_tweet = ' '.join(clean_tweet.split())
    clean_tweet = nltk.word_tokenize(clean_tweet)
    return clean_tweet
def lemmatization(tweet):
    # python -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # lemma_tweet = []
    # for i in tweets:
    #     t = [token.lemma_ for token in nlp(i)]
    #     lemma_tweet.append(' '.join(t))
    return [token.lemma_.lower() for token in nlp(tweet)]


def stem(tweet):
    tokenizer = nltk.RegexpTokenizer('[a-zA-Z0-9@]+')
    stemmer = nltk.LancasterStemmer()

    return [stemmer.stem(token) if not token.startswith('@') else token
            for token in tokenizer.tokenize(tweet)]