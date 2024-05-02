def tokenize(tokenizer, dataset):
    return tokenizer(dataset, padding="max_length", trunctaion=True)