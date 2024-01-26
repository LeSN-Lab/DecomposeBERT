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
def evaluate_model(model, test_dataloader, device):
    model.eval()

    # Initialize accuracy variables
    total_correct = 0
    total_count = 0
    total_eval_loss = 0
    print("Start testing")

    # Use the test_dataloader for evaluation
    for batch in tqdm(test_dataloader, desc="Evaluating"):

        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)

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
