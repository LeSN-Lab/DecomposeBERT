from utils.model_utils.load_model import *
from utils.data_utils.load_dataset import load_sdg
from modularization.concern_identification import *
from utils.model_utils.model_config import ModelConfig
from utils.decompose_utils.common import init_modular_layers, recurse_layers
from tqdm import tqdm

if __name__ == "__main__":
    model_dir = "SDGclassfier(gpt2-gpt)"
    model_name = "sadickam/sdg-classification-bert"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_name = "best_model.pt"
    model_config = ModelConfig(
        _model_name=model_name,
        _model_type="berts",
        _checkpoint_name=checkpoint_name,
        _num_labels=16,
        _device=device,
    )
    model, tokenizer, checkpoint = load_classification_model(model_config)

    ci = ConcernIdentification()

    train_dataloader, valid_dataloader, test_dataloader = load_sdg(
        model_config, tokenizer, batch_size=32
    )

    for j in tqdm(range(model_config.num_labels)):
        print("#Module " + str(j) + " in progress....")

        model, tokenizer, checkpoint = load_classification_model(model_config)

        positiveConcern = init_modular_layers(model)
        negativeConcern = init_modular_layers(model)

        with torch.no_grad():
            b_input_ids = batch["input_ids"].to(model_config.device)
            b_attention_mask = batch["attention_mask"].to(model_config.device)
            b_labels = batch["labels"].to(model_config.device)
            for batch in tqdm(train_dataloader, desc="Concern Identification"):
                recurse_layers(model, ci.propagateThroughLayer)
