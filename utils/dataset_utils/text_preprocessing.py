# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=AFWlSsbZaRLc
import re
from functools import partial
from nltk.corpus import stopwords
import multiprocessing as mp
import spacy


# Initialize Spacy and stop words
#python -m spacy download en_core_web_lg

nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = re.sub("[^a-zA-Z ]", "", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    return text


def preprocess(text):
    """Cleans and optionally lemmatizes the text, then tokenizes it using BERT tokenizer."""
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)
    cleaned_text = " ".join(
        [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    )
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
