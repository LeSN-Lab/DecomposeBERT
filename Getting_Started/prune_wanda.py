import copy
import torch
from datetime import datetime
from utils.helper import ModelConfig, color_print
from utils.dataset_utils.load_dataset import (
    load_data,
)
from utils.model_utils.load_model import load_model
from utils.model_utils.evaluate import evaluate_model, get_sparsity
from utils.dataset_utils.sampling import SamplingDataset
from utils.prune_utils.prune import (
    prune_wanda
)

name= "OSDG"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = None
model_config = ModelConfig(name, device)
num_labels = model_config.config["num_labels"]

for i in range(num_labels):
    model, tokenizer, checkpoint = load_model(model_config)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        name, batch_size=32
    )

    color_print("Start Time:" + datetime.now().strftime("%H:%M:%S"))
    color_print("#Module " + str(i) + " in progress....")
    num_samples = 64

    positive_samples = SamplingDataset(
        train_dataloader, i, num_samples, num_labels, True, 4, device=device
    )
    negative_samples = SamplingDataset(
        train_dataloader, i, num_samples, num_labels, False, 4, device=device
    )
    all_samples = SamplingDataset(
        train_dataloader, 200, 20, num_labels, False, 4, device=device
    )

    print("origin")
    # evaluate_model(model, model_config, test_dataloader)

    module = copy.deepcopy(model)
    prune_wanda(model, all_samples, sparsity_ratio=0.4)

    print(get_sparsity(module)[0])

    result = evaluate_model(module, model_config, test_dataloader)


