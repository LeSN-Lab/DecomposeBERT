# In[]
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.modelConfig import load_model
from utils.load_dataset import load_sdg
import torch
from tqdm import tqdm


# In[]: Train model
def train_model(model_name, load_path, device, epochs=3, checkpoint_path=None, test=True):
    model, tokenizer = load_model(model_name, load_path)
    model = model.to(device)

    # In[] : Load model
    trainDataloader, valDataloader, testDataloader = load_sdg(tokenizer, test_size=0.25, batch_size=128)

    # In[] : Set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(trainDataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    start_epoch = 0

    if not os.path.isdir('Models'):
        os.mkdir('Models')

    if checkpoint_path and os.path.isfile(os.path.join('Models', checkpoint_path)):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")

    best_val_loss = np.inf
    no_improve_epochs = 0
    early_stopping_threshold = 2

    # In[]: Training loop
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0

            progress_bar = tqdm(trainDataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, disable=False)
            for batch in progress_bar:
                # Load batch data in GPU
                texts, input_ids, attention_masks, labels = batch
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)

                # Initialize gradient to 0
                model.zero_grad()

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss

                # Backward propagation
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update optimizer and scheduler
                optimizer.step()
                scheduler.step()

                # Update loss
                total_loss += loss.item()
                avg_loss = total_loss / (len(trainDataloader))
                progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            # In[]: Validation
            val_accuracy, val_loss = evaluate_model(model, valDataloader, device)
            print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_threshold:
                    print("Early stopping triggered.")
                    break

            # In[]: Save model
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, os.path.join('Models', f"epoch_{epoch + 1}.pt"))

        if test:
            evaluate_model(model, testDataloader, device)
    except Exception as e:
        print(f"An error occurred: {e}")
    return model


# In[]: Test model
# Calculate accuracy
def flat_accuracy(preds, label_indices, label_list):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = [label_list[index] for index in label_indices.flatten()]

    correct_predictions = 0
    for pred, true_label in zip(pred_flat, labels_flat):
        if label_list[pred] == true_label:
            correct_predictions += 1

    return correct_predictions / len(labels_flat)


def evaluate_model(model, testDataloader, device):
    model.eval()

    # Initialize accuracy variables
    total_correct = 0
    total_count = 0
    total_eval_loss = 0
    print("Start testing")

    # Use the test_dataloader for evaluation
    for batch in tqdm(testDataloader, desc="Evaluating"):
        if batch is None:
            continue

        b_texts, b_input_ids, b_attention_mask, b_labels = batch  # Extract only the necessary tensors
        b_input_ids = b_input_ids.to(device)
        b_attention_mask = b_attention_mask.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)

        logits = outputs.logits
        loss = outputs.loss
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels_ids = b_labels.cpu().numpy()
        preds = np.argmax(logits, axis=1)
        total_correct += np.sum(preds == labels_ids)
        total_count += labels_ids.shape[0]

    avg_accuracy = total_correct / total_count
    avg_loss = total_eval_loss / len(testDataloader)

    print(f"Accuracy: {avg_accuracy:.2f}")
    print(f"Validation Loss: {avg_loss:.2f}")
    return avg_accuracy, avg_loss


if __name__ == '__main__':
    # In[]: Load model and datasets
    file = os.path.realpath('__file__')
    root = os.path.dirname(file)
    model_name = "sadickam/sdg-classification-bert"
    load_path = os.path.join(root, "SDGclassfierModelConfig")
    device = torch.device('cuda:0' if not torch.cuda.is_available() else 'cpu')
    train_model(model_name, load_path, device, epochs=10, checkpoint_path=None, test=True)
