# In[]
import numpy as np
import pandas as pd
import regex as re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import concurrent.futures

# nltk.download('punkt')


# In[]
def prep_text(text):
    """
    function for preprocessing text
    """

    # remove trailing characters (\s\n) and convert to lowercase
    clean_sents = []  # append clean con sentences
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [
            str(word_token).strip().lower() for word_token in sent_token.split()
        ]
        # word_tokens = [word_token for word_token in word_tokens if word_token not in punctuations]
        clean_sents.append(" ".join((word_tokens)))
    joined = " ".join(clean_sents).strip(" ")
    joined = re.sub(r"`", "", joined)
    joined = re.sub(r'"', "", joined)
    return joined


# In[]
import os

os.chdir("/bert_models/SDGclassfierBERT/MultilingualModelConfig/")
model_checkpoint = "./"
model_id = "sadickam/sdg-classification-MultilingualModelConfig"
label_list = [
    "sdg_1",
    "sdg_2",
    "sdg_3",
    "sdg_4",
    "sdg_5",
    "sdg_6",
    "sdg_7",
    "sdg_8",
    "sdg_9",
    "sdg_10",
    "sdg_11",
    "sdg_12",
    "sdg_13",
    "sdg_14",
    "sdg_15",
    "sdg_16",
]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# In[]
df = pd.read_csv("./osdg-dataset.csv", delimiter="\t")
df.drop(
    axis=1,
    labels=["doi", "text_id", "labels_negative", "labels_positive", "agreement"],
    inplace=True,
)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=111)
true_labels = []
predicted_labels = []


def process_row(row_tuple):
    raw_text = row_tuple.text
    text = prep_text(raw_text)
    true_label = row_tuple.sdg

    tokenized_text = tokenizer(text, return_tensors="pt")

    text_logits = model(**tokenized_text).logits
    predictions = torch.softmax(text_logits, dim=1).tolist()[0]
    predicted_label_index = predictions.index(max(predictions))

    return true_label, predicted_label_index


# In[]
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(
        tqdm(
            executor.map(process_row, test_df.itertuples(index=False), chunksize=1),
            total=len(test_df),
        )
    )

for true_label, predicted_label in results:
    true_labels.append(true_label)
    predicted_labels.append(predicted_label + 1)
# In[]
accuracy = accuracy_score(true_labels, predicted_labels)
print(true_labels)
print(predicted_labels)
print(f"Test Accuracy: {accuracy}")

# In[]
import numpy as np

tmp = np.asarray(test_df["text"])

for i in range(100):
    if true_labels[i] != predicted_labels[i]:
        print(f"{true_labels[i]} {predicted_labels[i]}")
        print(tmp[i])
