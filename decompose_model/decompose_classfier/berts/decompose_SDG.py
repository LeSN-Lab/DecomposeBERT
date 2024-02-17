from utils.model_utils.load_model import *
from utils.data_utils.load_dataset import load_dataset
from utils.model_utils.model_config import ModelConfig
from tqdm import tqdm
from transformers import AutoConfig
from collections import defaultdict

if __name__ == "__main__":
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

    # ci = ConcernIdentification()

    train_dataloader, valid_dataloader, test_dataloader = load_dataset(
        model_config, tokenizer, batch_size=32
    )

    ids = defaultdict(list)
    attention_masks = defaultdict(list)
    labels = defaultdict(list)

    for batch in train_dataloader:
        b_input_ids = batch["input_ids"].to(model_config.device)
        b_attention_masks = batch["attention_mask"].to(model_config.device)
        b_labels = batch["labels"].to(model_config.device)

        for i in range(len(b_labels)):
            label = b_labels[i].item()

            ids[label].append(b_input_ids[i].cpu().tolist())
            attention_masks[label].append(b_attention_masks[i].cpu().tolist())
            labels[label].append(label)

    for j in tqdm(range(num_labels)):
        print("#Module " + str(j) + " in progress....")
    #
    #     model, tokenizer, checkpoint = load_classification_model(model_config)
    #
    #     positiveConcern = init_modular_layers(model)
    #     negativeConcern = init_modular_layers(model)
    #

        ids_tensor = torch.tensor(ids[label], dtype=torch.long).to(model_config.device)
        attention_masks_tensor = torch.tensor(attention_masks[label], dtype=torch.long).to(model_config.device)
        labels_tensor = torch.tensor(labels[label], dtype=torch.long).to(model_config.device)

    #     with torch.no_grad():
    #         b_input_ids = batch["input_ids"].to(model_config.device)
    #         b_attention_mask = batch["attention_mask"].to(model_config.device)
    #         b_labels = batch["labels"].to(model_config.device)
    #         for batch in tqdm(train_dataloader, desc="Concern Identification"):
    #             recurse_layers(model, ci.propagateThroughLayer)
