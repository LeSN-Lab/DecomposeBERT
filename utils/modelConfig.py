# in[] Library
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[] Load/Save MultilingualModelConfig
def save_model(model_name, save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(model)
    return model, tokenizer


def load_model(model_name, load_path):
    model = AutoModelForSequenceClassification.from_pretrained(load_path)
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    print(model)
    return model, tokenizer