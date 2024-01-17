# in[] Library
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# In[] Load/Save model_config
save_path = "utils/SDGclassfierBertmodel_config/model_config"
if not os.path.isdir(save_path):
    tokenizer = AutoTokenizer.from_pretrained("sadickam/sdg-classification-bert")
    model = AutoModelForSequenceClassification.from_pretrained("sadickam/sdg-classification-bert")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
else:

    tokenizers = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)

# In[]
print(model)