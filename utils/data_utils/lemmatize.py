from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


# Function to lemmatize a single word
def lemmatize_word(word):
    return lemmatizer.lemmatize(word)
