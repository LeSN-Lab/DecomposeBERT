import os
import torch
from utils.model_utils.load_model import load_model
from utils.model_utils.common import initModularLayers
from utils.enums import getLayerType

file = os.path.realpath("__file__")
root = os.path.dirname(file)
model_name = "sadickam/sdg-classification-bert"
load_path = os.path.join(root, "SDGclassfierModelConfig")

checkpoint_path = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, tokenizer, checkpoint = load_model(model_name, load_path, checkpoint_path)
model = model.to(device)
print(model.bert)
layers = initModularLayers(model)
# 토큰 임베딩 가중치
print(getLayerType(layers['input_embedding']))
