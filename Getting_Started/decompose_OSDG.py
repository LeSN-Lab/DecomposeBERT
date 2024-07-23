import copy
import torch
from datetime import datetime
from utils.helper import ModelConfig, color_print
from utils.dataset_utils.load_dataset import (
    load_data, convert_dataset_labels_to_binary,
)
from utils.model_utils.load_model import load_model
from utils.model_utils.evaluate import evaluate_model, get_sparsity
from utils.model_utils.save_module import save_module
from utils.decompose_utils.weight_remover import WeightRemoverBert
from utils.decompose_utils.concern_identification import ConcernIdentificationBert
from utils.decompose_utils.tangling_identification import TanglingIdentification
from utils.decompose_utils.concern_modularization import ConcernModularizationBert
from utils.decompose_utils.sampling import sampling_class
from utils.prune_utils.prune import prune_concern_identification, prune_magnitude

name= "OSDG"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = None
model_config = ModelConfig(name, device)
num_labels = model_config.config["num_labels"]

for i in range(num_labels):
    model, tokenizer, checkpoint = load_model(model_config)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        name, batch_size=64
    )

    color_print("Start Time:" + datetime.now().strftime("%H:%M:%S"))
    color_print("#Module " + str(i) + " in progress....")
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

    print("origin")
    # evaluate_model(model, model_config, test_dataloader)

    module = copy.deepcopy(model)
    wr = WeightRemoverBert(model, p=0.9)
    ci = ConcernIdentificationBert(model, p=0.4)
    ti = TanglingIdentification(model, p=0.5)

    print("Start Positive CI sparse")
    prune_magnitude(module, sparsity_ratio=0.1)
    print(get_sparsity(module)[0])
    print("Start Positive CI after sparse")

    prune_concern_identification(model, module, positive_samples, include_layers=["attention", "intermediate", "output"], sparsity_ratio=0.6)

    # for idx, batch in enumerate(positive_samples):
    #     input_ids, attn_mask, _, total_sampled = batch
    #     with torch.no_grad():
    #         ci.propagate(module, input_ids)
        # if idx % eval_step:
    print(get_sparsity(module))

    result = evaluate_model(module, model_config, test_dataloader)

    print("Start Negative TI")

    for idx, batch in enumerate(negative_samples):
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            ti.propagate(module, input_ids)
        # if idx % eval_step:
        #     evaluate_model(module, model_config, test_dataloader)
    result = evaluate_model(module, model_config, test_dataloader)
    print(get_sparsity(module))
