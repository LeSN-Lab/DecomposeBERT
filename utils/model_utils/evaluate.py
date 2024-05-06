# In[]: Import Libraries
import torch
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import classification_report, precision_recall_fscore_support


def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).sum().item() / len(labels)
    return {"accuracy": accuracy}


def evaluate_model(model, model_config, test_dataloader, is_multi_label=True):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    if is_multi_label:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(model_config.device)
        attention_mask = batch["attention_mask"].to(model_config.device)
        labels = batch["labels"].to(model_config.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if is_multi_label:
                labels = F.one_hot(labels, num_classes=model.num_labels).float()
                loss = criterion(logits, labels.float())
                pred = torch.sigmoid(logits) > 0.5
            else:
                loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                pred = logits.argmax(dim=1)

            total_loss += loss.mean().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)
    if is_multi_label:
        report = classification_report(all_labels, all_preds, zero_division=0)
        print(f"Loss: {avg_loss:.4f}")
        print(report)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
        print(f"Loss: {avg_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    print(f"Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return {
        "loss": avg_loss,
        "report": report if is_multi_label else None,
        "precision": precision if not is_multi_label else None,
        "recall": recall if not is_multi_label else None,
        "f1_score": f1 if not is_multi_label else None,
    }


def test_f1(module, test_dataloader, model_config, is_binary=True, concern_num=0):
    all_preds = []
    all_true = []

    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(model_config.device)
        attn_mask = batch["attention_mask"].to(model_config.device)
        true_labels = batch["labels"].to(model_config.device)

        with torch.no_grad():
            output = module(input_ids, attn_mask)
        if is_binary:
            preds = (output.logits.squeeze() > 0).long()
            true_labels = (true_labels == concern_num).long()
        else:
            preds = output.logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(true_labels.cpu().numpy())

    if is_binary:
        report = classification_report(
            all_true,
            all_preds,
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            output_dict=True,
            zero_division=0,
        )
    else:
        report = classification_report(
            all_true, all_preds, output_dict=True, zero_division=0
        )

    if is_binary:
        detailed_metrics = report["Positive"]  # Or '0' for the negative class
        return {"detailed_metrics": detailed_metrics, "details": report}
    else:
        macro_f1 = report["macro avg"]["f1-score"]
        return {"macro_f1": macro_f1, "details": report}
