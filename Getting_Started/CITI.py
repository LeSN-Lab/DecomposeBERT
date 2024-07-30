import os
import sys

sys.path.append("../")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
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
    prune_magnitude,
    prune_concern_identification,
    recover_tangling_identification,
)


def main():
    parser = argparse.ArgumentParser(description="Model Pruning and Evaluation")
    parser.add_argument("--name", type=str, default="OSDG", help="Name of the dataset")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for computation"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load model from"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for data loaders"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for data loaders"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples for sampling dataset",
    )
    parser.add_argument(
        "--concern", type=int, default=0, help="Target Concern for decompose"
    )
    parser.add_argument(
        "--magnitude_ratio",
        type=float,
        default=0.1,
        help="Sparsity ratio for magnitude pruning",
    )
    parser.add_argument(
        "--ci_ratio",
        type=float,
        default=0.6,
        help="Sparsity ratio for concern identification pruning",
    )
    parser.add_argument(
        "--ti_ratio",
        type=float,
        default=0.1,
        help="Recovery ratio for tangling identification",
    )
    parser.add_argument(
        "--include_layers",
        type=str,
        nargs="+",
        default=["attention", "intermediate", "output"],
        help="Layers to include for pruning",
    )
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="+",
        default=None,
        help="Layers to exclude for pruning",
    )
    parser.add_argument("--log_dir", type=str, help="Path to the log file.")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_config = ModelConfig(args.name, device)
    num_labels = model_config.config["num_labels"]

    i = args.concern

    if args.log_dir:
        sys.stdout = open(f"Logs/{args.log_dir}", 'a')
        sys.stderr = open(f"Logs/{args.log_dir}", 'a')

    color_print("Start Time:" + datetime.now().strftime("%H:%M:%S"))
    model, tokenizer, checkpoint = load_model(model_config)

    train_dataloader, valid_dataloader, test_dataloader = load_data(
        args.name, batch_size=args.batch_size, num_workers=args.num_workers
    )

    color_print("#Module " + str(i) + " in progress....")

    positive_samples = SamplingDataset(
        train_dataloader, i, args.num_samples, num_labels, True, 4, device=device
    )
    negative_samples = SamplingDataset(
        train_dataloader, i, args.num_samples, num_labels, False, 4, device=device
    )
    all_samples = SamplingDataset(
        train_dataloader,
        10000,
        args.num_samples,
        num_labels,
        False,
        4,
        device=device,
    )

    print("Evaluate the original model")
    # result = evaluate_model(model, model_config, test_dataloader)

    module = copy.deepcopy(model)
    prune_magnitude(
        module,
        include_layers=args.include_layers,
        exclude_layers=args.exclude_layers,
        sparsity_ratio=args.magnitude_ratio,
    )
    print(get_sparsity(module)[0])

    prune_concern_identification(
        model,
        module,
        positive_samples,
        include_layers=args.include_layers,
        exclude_layers=args.exclude_layers,
        sparsity_ratio=args.ci_ratio,
    )

    print(get_sparsity(module)[0])
    # result = evaluate_model(module, model_config, test_dataloader)
    recover_tangling_identification(
        model,
        module,
        negative_samples,
        include_layers=args.include_layers,
        exclude_layers=args.exclude_layers,
        recovery_ratio=args.ti_ratio,
    )
    print(get_sparsity(module)[0])
    result = evaluate_model(module, model_config, test_dataloader)
    # save_module(module, "Modules/", "module.pt")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
