# %%
import copy
import os.path
import sys

# %%
pwd = os.getcwd()
# %%
sys.path.append(os.path.dirname(pwd))
# %%
from utils.model_utils.evaluate import evaluate_model, test_f1, test

from utils.model_utils.load_model import *
from utils.model_utils.model_config import ModelConfig
from utils.dataset_utils.load_dataset import load_dataset
from utils.decompose_utils.weight_remover import WeightRemoverBert
from utils.decompose_utils.concern_identification import ConcernIdentificationBert
from utils.decompose_utils.tangling_identification import TanglingIdentification
from transformers import AutoConfig
from tqdm import tqdm
from datetime import datetime
from pprint import pp
from utils.decompose_utils.sampling import sampling_class
import torch


# %%
def save_module(module, module_num):
    save_path = f"{model_config.module_dir}/{module_num}.pt"
    torch.save(module.state_dict(), save_path)


# %%
# model_name = "prajjwal1/bert-tiny"
# model_dir = "tiny bert"
model_name = "bert-base-uncased"
model_dir = "bert"
data = "SDG"
num_labels = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_name = "best_model.pt"
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model_config = ModelConfig(
    _model_name=model_name,
    _model_dir=model_dir,
    _data=data,
    _is_pretrained=True,
    _transformer_config=config,
    _checkpoint_name=checkpoint_name,
    _device=device,
)
model, tokenizer, checkpoint = load_classification_model(model_config)

train_dataloader, valid_dataloader, test_dataloader = load_dataset(
    model_config, tokenizer, batch_size=32, test_size=0.3
)
# print("Start Time:" + datetime.now().strftime("%H:%M:%S"))
# evaluate_model(model, model_config, test_dataloader)
# %%
i = 4
print("#Module " + str(i) + " in progress....")
num_samples = 64

positive_samples1 = sampling_class(
    train_dataloader, i, 20, num_labels, True, 4, device=device
)
positive_samples2 = sampling_class(
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
results = test_f1(model, test_dataloader, model_config, False)
pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

print("Start Positive CI after sparse")

for batch in positive_samples2:
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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

for num in range(config.num_hidden_layers):
    print(f"f({ff1[num]})")
    print("\n")
for num in range(config.num_hidden_layers):
    print(f"f({ff2[num]})")
    print("\n")
print(f"f({pooler})")
print(f"f({classifier})")

save_module(module1, i)
# %%
i = 5
print("#Module " + str(i) + " in progress....")
num_samples = 64

positive_samples1 = sampling_class(
    train_dataloader, i, 20, num_labels, True, 4, device=device
)
positive_samples2 = sampling_class(
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
results = test_f1(model, test_dataloader, model_config, False)
pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

print("Start Positive CI after sparse")

for batch in positive_samples2:
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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

for num in range(config.num_hidden_layers):
    print(f"f({ff1[num]})")
    print("\n")
for num in range(config.num_hidden_layers):
    print(f"f({ff2[num]})")
    print("\n")
print(f"f({pooler})")
print(f"f({classifier})")

save_module(module1, i)
# %%
i = 6
print("#Module " + str(i) + " in progress....")
num_samples = 64

positive_samples1 = sampling_class(
    train_dataloader, i, 20, num_labels, True, 4, device=device
)
positive_samples2 = sampling_class(
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
results = test_f1(model, test_dataloader, model_config, False)
pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

print("Start Positive CI after sparse")

for batch in positive_samples2:
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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

for num in range(config.num_hidden_layers):
    print(f"f({ff1[num]})")
    print("\n")
for num in range(config.num_hidden_layers):
    print(f"f({ff2[num]})")
    print("\n")
print(f"f({pooler})")
print(f"f({classifier})")

save_module(module1, i)
# %%
i = 14
print("#Module " + str(i) + " in progress....")
num_samples = 64

positive_samples1 = sampling_class(
    train_dataloader, i, 20, num_labels, True, 4, device=device
)
positive_samples2 = sampling_class(
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
results = test_f1(model, test_dataloader, model_config, False)
pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

print("Start Positive CI after sparse")

for batch in positive_samples2:
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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

for num in range(config.num_hidden_layers):
    print(f"f({ff1[num]})")
    print("\n")
for num in range(config.num_hidden_layers):
    print(f"f({ff2[num]})")
    print("\n")
print(f"f({pooler})")
print(f"f({classifier})")

save_module(module1, i)
# %%
i = 15
print("#Module " + str(i) + " in progress....")
num_samples = 64

positive_samples1 = sampling_class(
    train_dataloader, i, 20, num_labels, True, 4, device=device
)
positive_samples2 = sampling_class(
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
results = test_f1(model, test_dataloader, model_config, False)
pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

print("Start Positive CI after sparse")

for batch in positive_samples2:
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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

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
            torch.sum(module1.bert.encoder.layer[num].output.dense.weight != 0).item()
        )
    pooler.append(torch.sum(module1.bert.pooler.dense.weight != 0).item())
    classifier.append(torch.sum(module1.classifier.weight != 0).item())

    j += 1
    print(j)

    results = test_f1(module1, test_dataloader, model_config, False)
    pp(results["details"][f"{i}"])

for num in range(config.num_hidden_layers):
    print(f"f({ff1[num]})")
    print("\n")
for num in range(config.num_hidden_layers):
    print(f"f({ff2[num]})")
    print("\n")
print(f"f({pooler})")
print(f"f({classifier})")

save_module(module1, i)
