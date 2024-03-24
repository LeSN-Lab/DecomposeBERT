# In[]: Import Libraries
import numpy as np
import torch
from tqdm import tqdm


# In[]: Test model
# Calculate accuracy
def flat_accuracy(pred, label_indices, label_list):
    pred_flat = np.argmax(pred, axis=1).flatten()
    labels_flat = [label_list[index] for index in label_indices.flatten()]

    correct_predictions = 0
    for _pred, true_label in zip(pred_flat, labels_flat):
        if label_list[_pred] == true_label:
            correct_predictions += 1

    return correct_predictions / len(labels_flat)


# In[]
def evaluate_model(model, model_config, test_dataloader):
    model.eval()

    # Initialize accuracy variables
    total_correct = 0
    total_count = 0
    total_eval_loss = 0
    print("\nStart testing")

    # Use the test_dataloader for evaluation
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        b_input_ids = batch["input_ids"].to(model_config.device)
        b_attention_mask = batch["attention_mask"].to(model_config.device)
        b_labels = batch["labels"].to(model_config.device)

        with torch.no_grad():
            outputs = model(
                b_input_ids, attention_mask=b_attention_mask, labels=b_labels
            )

        logits = outputs.logits
        loss = outputs.loss
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels_ids = b_labels.cpu().numpy()
        pred = np.argmax(logits, axis=1)
        total_correct += np.sum(pred == labels_ids)
        total_count += labels_ids.shape[0]

    avg_accuracy = total_correct / total_count
    avg_loss = total_eval_loss / len(test_dataloader)

    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_accuracy, avg_loss


def test_f1(module, test_dataloader, model_config, test_class):
    TP, FP, FN = 0, 0, 0

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(model_config.device)
        attn_mask = batch['attention_mask'].to(model_config.device)
        true_labels = batch['labels'].to(model_config.device)

        with torch.no_grad():
            output = module(input_ids, attn_mask)
            preds = output.logits.argmax(dim=-1)

            for pred, true_label in zip(preds, true_labels):
                if pred == true_label and true_label == test_class:
                    TP += 1
                elif pred != true_label and true_label == test_class:
                    FN += 1
                elif pred == test_class and true_label != test_class:
                    FP += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def test(module, test_ids, test_labels, test_mask, test_case):
    cnt = 0
    correct = {}
    for j in range(test_case):
        input_ids = test_ids[j].unsqueeze(0)
        attn_mask = test_mask[j].unsqueeze(0)
        with torch.no_grad():
            output = module(input_ids, attn_mask)
            pred = output.logits.argmax(dim=1).item()
            if pred == test_labels[j].item():
                cnt += 1

    accuracy = cnt / test_case
    return accuracy