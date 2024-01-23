# In[]
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# In[]
# Load model directly
model_name = "nlptown/MultilingualModelConfig-base-multilingual-uncased-sentiment"
save_path = 'BERTS/Multilingual/MultilingualModelConfig'
print(save_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# In[]
print(model)

