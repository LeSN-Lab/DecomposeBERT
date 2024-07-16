import copy
from utils.model_utils.evaluate import evaluate_model
from utils.model_utils.load_model import load_model
from utils.model_utils.model_config import ModelConfig
from utils.dataset_utils.load_dataset import load_data
from utils.decompose_utils.weight_remover import WeightRemoverBert
from utils.decompose_utils.concern_identification import ConcernIdentificationBert
from utils.decompose_utils.tangling_identification import TanglingIdentification
from utils.model_utils.save_module import save_module
from datetime import datetime
from utils.decompose_utils.concern_modularization import ConcernModularizationBert
from utils.decompose_utils.sampling import sampling_class
from utils.dataset_utils.load_dataset import convert_dataset_labels_to_binary, extract_and_convert_dataloader
import torch

model_name = "textattack/bert-base-uncased-imdb"
task_type = "classification"
architectures = "bert"
dataset_name = "IMDb"
num_labels = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = None
model_config = ModelConfig(
    model_name=model_name,
    task_type=task_type,
    dataset_name=dataset_name,
    checkpoint=checkpoint,
    device=device,
)

for i in range(num_labels):
    model, tokenizer, checkpoint = load_model(model_config)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        model_config, batch_size=32
    )

    print("Start Time:" + datetime.now().strftime("%H:%M:%S"))
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

    print("origin")
    evaluate_model(model, model_config, test_dataloader)

    module = copy.deepcopy(model)
    wr = WeightRemoverBert(model, p=0.9)
    ci = ConcernIdentificationBert(model, p=0.4)
    ti = TanglingIdentification(model, p=0.5)

    print("Start Positive CI sparse")

    eval_step = 5
    for idx, batch in enumerate(all_samples):
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            wr.propagate(module, input_ids)
        if idx % eval_step:
            evaluate_model(module, model_config, test_dataloader)

    print("Start Positive CI after sparse")

    for idx, batch in enumerate(positive_samples):
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            ci.propagate(module, input_ids)
        if idx % eval_step:
            evaluate_model(module, model_config, test_dataloader)

    print("Start Negative TI")

    for idx, batch in enumerate(negative_samples):
        input_ids, attn_mask, _, total_sampled = batch
        with torch.no_grad():
            ti.propagate(module, input_ids)
        if idx % eval_step:
            evaluate_model(module, model_config, test_dataloader)

    ConcernModularizationBert.channeling(module, ci.active_node, ti.dead_node, i, model_config.device)
    binary_module = ConcernModularizationBert.convert2binary(model_config, module)

    converted_test_dataloader = convert_dataset_labels_to_binary(test_dataloader, i, True)
    result = evaluate_model(binary_module, model_config, converted_test_dataloader)
    save_module(binary_module, model_config.module_dir, model_config.model_name)
