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

# In[] print model_config state_dict
def categorize_and_print_layers():
    embedding_layers = []
    attention_layers = []
    layer_norm_layers = []
    linear_layers = []
    other_layers = []

    for name, param in model.named_parameters():
        if "embeddings" in name:
            embedding_layers.append(name)
        elif "attention" in name:
            attention_layers.append(name)
        elif "LayerNorm" in name:
            layer_norm_layers.append(name)
        elif "linear" in name or "dense" in name:
            linear_layers.append(name)
        else:
            other_layers.append(name)

    # Print each category
    print("Embedding Layers:\n", embedding_layers, "\n")
    print("Attention Layers:\n", attention_layers, "\n")
    print("Layer Normalization Layers:\n", layer_norm_layers, "\n")
    print("Linear (Dense) Layers:\n", linear_layers, "\n")
    print("Other Layers:\n", other_layers, "\n")

categorize_and_print_layers()
# In[]
print(model.summary())