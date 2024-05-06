def tokenize(tokenizer, dataset):
    tokenized_data = []
    for sample in dataset:
        encoded_input = tokenizer(
            sample['text'],
            padding='max_length',  # or use other padding strategy
            truncation=True,
            return_tensors='pt'
        )
        tokenized_data.append({
            'input_ids': encoded_input['input_ids'].squeeze(0),
            'attention_mask': encoded_input['attention_mask'].squeeze(0),
            'label': sample['label']  # Assuming you have a label key in the sample
        })
    return tokenized_data