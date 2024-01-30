import spacy
import re
import nltk




def saveTFDataset(datasetName, trainSize=100):
    X_train, y_train = \
        tfds.as_numpy(tfds.load(datasetName,
                                split='train[:' + str(trainSize) + '%]',
                                batch_size=-1,
                                as_supervised=True))

    out = open("cleaned_sentiment140.csv", "w")
    out.write("Twwet,Sentiment\n")

    taken = np.array([0] * 5)
    for index, tweet in enumerate(X_train):
        if np.sum(taken) >= 10:
            break
        if taken[y_train[index]] > 2:
            continue

        taken[y_train[index]] += 1

        tweet = lemmatization(text_processing(tweet))
        tweet = " ".join(tweet)
        tweet = tweet.replace(".", " ")
        tweet = tweet.replace("\\", "")
        tweet = tweet.replace("'", "")
        tweet = tweet.replace('"', "")
        out.write('"' + tweet + '",' + str(y_train[index]) + "\n")
