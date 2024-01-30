import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spacy.lang.en import English
import multiprocessing as mp


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
nlp = English()


def clean_text(text):
  """Cleans text by removing special characters, URLs, email addresses, and extra whitespace."""
  text = re.sub(r'\s+', ' ', text).strip()
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'http\S+', '', text)
  text = re.sub(r'\S+@\S+\s?', '', text)
  return text

def lemmatization(sentences):
  """Lemmatizes words in the sentences."""
  lemmatized_sentences = []
  with mp.Pool() as pool:
    lemmatized_sentences = pool.map(lemmatizer, sentences)
  return lemmatized_sentences

def remove_stopwords(sentences):
  """Removes stopwords from the sentences."""
  filtered_sentences = []
  for sentence in sentences:
    filtered_words = [token for token in nlp(sentence) if token.is_stop == False]
    filtered_sentence = ' '.join(filtered_words)
    filtered_sentences.append(filtered_sentence)
  return filtered_sentences


def pre_processing(text):
  """Preprocesses text by cleaning, lemmatizing, and removing stopwords."""
  sentences = clean_text(text)
  lemmatized_sentences = lemmatization(sentences)
  processed_sentences = remove_stopwords(lemmatized_sentences)

  unique_words = set(" ".join(processed_sentences).split())
  return unique_words