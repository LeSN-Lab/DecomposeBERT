from utils.model_utils.load_model import *
from utils.data_utils.load_dataset import load_sdg
from modularization.concern_identification import *
from utils.model_utils.model_config import ModelConfig
from utils.decompose_utils.common import init_modular_layers

if __name__ == "__main__":
    model_dir = "SDGclassfier(pre_trained)"
    model_name = "sadickam/sdg-classification-bert"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_name = "best_model.pt"
    model_config = ModelConfig(
        _model_dir=model_dir,
        _model_name=model_name,
        _model_type="berts",
        _checkpoint_name=checkpoint_name,
        _num_labels=16,
        _device=device,
    )
    model, tokenizer, checkpoint = load_classification_model(model_config)

    ci = ConcernIdentification()

    # train_dataloader, valid_dataloader, test_dataloader = load_sdg(
    #     model_config, tokenizer, batch_size=32
    # )

    for j in range(model_config.num_labels):
        print("#Module " + str(j) + " in progress....")

        model, tokenizer, checkpoint = load_classification_model(model_config)

        layers = init_modular_layers(model)
