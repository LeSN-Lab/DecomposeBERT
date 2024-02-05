import re
from functools import partial
from nltk.corpus import stopwords
import multiprocessing as mp
import spacy


# Initialize Spacy and stop words
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.replace('[^a-zA-Z ]', '')
    text = text.replace('^ +', "")
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\S+@\S+\s?", "", text)  # Remove emails

    return text.lower()


def preprocess(text):
    """Cleans and optionally lemmatizes the text, then tokenizes it using BERT tokenizer."""
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)
    cleaned_text = ' '.join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
    return cleaned_text.lower()


def worker_init():
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess_wrapper(text):
    return preprocess(text)


def preprocess_texts(text_list):
    """Preprocesses a list of texts in parallel using multiprocessing."""
    # Define a wrapper function for starmap to include tokenizer in the call
    preprocess_with_args = partial(preprocess_wrapper)

    with mp.Pool(mp.cpu_count(), initializer=worker_init) as pool:
        result = pool.map(preprocess_with_args, text_list)
    return result
