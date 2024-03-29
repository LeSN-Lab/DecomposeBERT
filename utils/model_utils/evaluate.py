# In[]: Import Libraries
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report


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

def test_f1(module, test_dataloader, model_config, is_binary=True):
    all_preds = []
    all_true = []

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(model_config.device)
        attn_mask = batch['attention_mask'].to(model_config.device)
        true_labels = batch['labels'].to(model_config.device)

        with torch.no_grad():
            output = module(input_ids, attn_mask)
            preds = output.logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(true_labels.cpu().numpy())

    report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)

    if is_binary:
        detailed_metrics = report['1']  # Or '0' for the negative class
        return {'detailed_metrics': detailed_metrics, 'classification_report': report}
    else:
        macro_f1 = report['macro avg']['f1-score']
        return {"macro_f1": macro_f1, "details": report}

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