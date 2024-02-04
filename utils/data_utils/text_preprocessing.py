import re
from nltk.corpus import stopwords
import multiprocessing as mp
import spacy

# Initialize lemmatizer and stop words
stop_words = set(stopwords.words("english"))
pool = mp.Pool(mp.cpu_count())
nlp = spacy.load("en_core_web_sm")

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words


def clean_text(text):
  text = re.sub(r'\s+', ' ', text).strip()
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'http\S+', '', text)
  text = re.sub(r'\S+@\S+\s?', '', text)
  return text.lower()


def pre_processing(text):
  # Clean the text first
  cleaned_text = clean_text(text)
  doc = nlp(cleaned_text)
  lemmatized_text = ' '.join([token.lemma_ for token in doc if token.lemma_ not in spacy_stop_words])
  return lemmatized_text
