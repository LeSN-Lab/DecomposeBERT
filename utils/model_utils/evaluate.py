# In[]: Import Libraries
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from utils.prune_utils.prune import find_layers


def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).sum().item() / len(labels)
    return {"accuracy": accuracy}


def evaluate_model(model, model_config, test_dataloader, is_binary=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(test_dataloader, desc="Evaluating", dynamic_ncols=True, leave=True):
        input_ids = batch["input_ids"].to(model_config.device)
        attention_mask = batch["attention_mask"].to(model_config.device)
        labels = batch["labels"].to(model_config.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            pred = logits.argmax(dim=1)

            total_loss += loss.mean().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary" if is_binary else "macro", zero_division=0
    )

    report = classification_report(all_labels, all_preds, zero_division=0)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(report)

    return {
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report": report,
    }

def calculate_sparsity(param):
    return (param == 0).sum().item() / param.numel()

def get_sparsity(model, layer_types=None, include_layers=None, exclude_layers=None):
    layers = find_layers(model, layer_types=layer_types, include_layers=include_layers, exclude_layers=exclude_layers)
    sparsity_dict = {}
    total_sparsity = 0
    total_params = 0

    for name, module in layers.items():
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                sparsity = calculate_sparsity(param.data)
                sparsity_dict[f"{name}.{param_name}"] = sparsity
                total_sparsity += sparsity * param.numel()
                total_params += param.numel()

    overall_sparsity = total_sparsity / total_params if total_params != 0 else 0
    return overall_sparsity, sparsity_dict
    
    