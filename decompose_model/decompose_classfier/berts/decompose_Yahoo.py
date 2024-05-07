import copy
import os.path
import sys

pwd = os.getcwd()
sys.path.append(os.path.dirname(pwd))
from utils.model_utils.evaluate import evaluate_model
from utils.model_utils.load_model import *
from utils.model_utils.model_config import ModelConfig
from utils.dataset_utils.load_dataset import load_data
from utils.decompose_utils.weight_remover import WeightRemoverBert
from utils.decompose_utils.concern_identification import ConcernIdentificationBert
from utils.decompose_utils.tangling_identification import TanglingIdentification
from transformers import AutoConfig
from utils.model_utils.save_module import save_module
from datetime import datetime
from utils.decompose_utils.sampling import sampling_class
import torch


# model_name = "bert-base-uncased"
# model_type = "bert"

model_name = "fabriceyhc/bert-base-uncased-yahoo_answers_topics"
model_type = "pretrained"

data = "Yahoo"
num_labels = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_name = None
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model_config = ModelConfig(
    _model_name=model_name,
    _model_type=model_type,
    _data=data,
    _transformer_config=config,
    _checkpoint_name=checkpoint_name,
    _device=device,
)
model, tokenizer, checkpoint = load_classification_model(model_config, train_mode=False)

train_dataloader, valid_dataloader, test_dataloader = load_data(
    model_config, batch_size=32, test_size=0.3
)
print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

for i in range(num_labels):
    print("#Module " + str(i) + " in progress....")
    num_samples = 64

    positive_samples = sampling_class(
        train_dataloader, i, num_samples, num_labels, True, 4, device=device
    )
    negative_samples = sampling_class(
        train_dataloader, i, num_samples, num_labels, False, 4, device=device
    )

    all_samples = sampling_class(
        train_dataloader, 200, 20, num_labels, False, 4, device=device
    )

    module1 = copy.deepcopy(model)
    w = WeightRemoverBert(model, p=0.9)
    ci1 = ConcernIdentificationBert(model, p=0.4)
    ti1 = TanglingIdentification(model, p=0.6)

    ff1 = [
        [torch.sum(model.bert.encoder.layer[num].intermediate.dense.weight != 0).item()]
        for num in range(config.num_hidden_layers)
    ]
    ff2 = [
        [torch.sum(model.bert.encoder.layer[num].output.dense.weight != 0).item()]
        for num in range(config.num_hidden_layers)
    ]
    pooler = [torch.sum(model.bert.pooler.dense.weight != 0).item()]
    classifier = [torch.sum(model.classifier.weight != 0).item()]
    print("origin")
    j = 0
    print(j)
    # result = evaluate_model(model, model_config, test_dataloader)

    print("Start Positive CI sparse")

    for batch in all_samples:
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            t1 = w.propagate(module1, input_ids)
        for num in range(config.num_hidden_layers):
            ff1[num].append(
                torch.sum(
                    module1.bert.encoder.layer[num].intermediate.dense.weight != 0
                ).item()
            )
            ff2[num].append(
                torch.sum(
                    module1.bert.encoder.layer[num].output.dense.weight != 0
                ).item()
            )
        pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
        classifier.append(torch.sum(module1.classifier.weight != 0).item())

        j += 1
        print(j)

        # result = evaluate_model(module1, model_config, test_dataloader)

    print("Start Positive CI after sparse")

    for batch in positive_samples:
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            t1 = ci1.propagate(module1, input_ids)
        for num in range(config.num_hidden_layers):
            ff1[num].append(
                torch.sum(
                    module1.bert.encoder.layer[num].intermediate.dense.weight != 0
                ).item()
            )
            ff2[num].append(
                torch.sum(
                    module1.bert.encoder.layer[num].output.dense.weight != 0
                ).item()
            )
        pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
        classifier.append(torch.sum(module1.classifier.weight != 0).item())

        j += 1
        print(j)

        # result = evaluate_model(module1, model_config, test_dataloader)

    print("Start Negative TI")

    for batch in negative_samples:
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            t = ti1.propagate(module1, input_ids)
        for num in range(config.num_hidden_layers):
            ff1[num].append(
                torch.sum(
                    module1.bert.encoder.layer[num].intermediate.dense.weight != 0
                ).item()
            )
            ff2[num].append(
                torch.sum(
                    module1.bert.encoder.layer[num].output.dense.weight != 0
                ).item()
            )
        pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
        classifier.append(torch.sum(module1.classifier.weight != 0).item())

        j += 1
        print(j)

        # result = evaluate_model(module1, model_config, test_dataloader)
    result = evaluate_model(module1, model_config, test_dataloader)

    for num in range(config.num_hidden_layers):
        print(f"f({ff1[num]})")
        print("\n")
    for num in range(config.num_hidden_layers):
        print(f"f({ff2[num]})")
        print("\n")
    print(f"f({pooler})")
    print(f"f({classifier})")

    save_module(module1, model_config.module_dir, f"{i}.pt")
