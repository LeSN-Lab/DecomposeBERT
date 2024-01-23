# In[]
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.modelConfig import save_model, load_model
from utils.load_dataset import load_sdg
import torch
from tqdm import tqdm


# In[]: Train model
def train_model(model, tokenizer, device='cuda', root='./', checkpoint_path=None):
    # In[] : Load model
    train_dataloader, validation_dataloader = load_sdg(tokenizer)

    # In[] : Set optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3

    # In[] : Scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    start_epoch = 0

    # In[]: Training loop
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, disable=False)
        for batch_idx, batch in enumerate(progress_bar):
            # Load batch data in GPU
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            # Initialize gradient to 0
            model.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            # Backward propagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update optimizer and scheduler
            optimizer.step()
            scheduler.step()

            # Update loss
            loss_value = loss.item()
            total_loss += loss_value
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # In[]: Save model
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": total_loss / len(train_dataloader)
    }, os.path.join(root, f"checkpoint_epoch_{epoch + 1}.pt"))


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


def test_model(model, tokenizer, device):
    _, test_dataloader = load_sdg(tokenizer)
    model.to(device)
    model.eval()

    # Initialize accuracy variables
    total_correct = 0
    total_count = 0
    total_eval_loss = 0
    print("Start testing")

    # Use the test_dataloader for evaluation
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        b_texts, b_input_ids, b_labels = batch  # Extract only the necessary tensors
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)

        b_attention_mask = (b_input_ids != 0).type(torch.LongTensor).to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)

        logits = outputs.logits
        loss = outputs.loss
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels_ids = b_labels.cpu().numpy()
        preds = np.argmax(logits, axis=1) + 1
        total_correct += np.sum(preds == labels_ids)
        total_count += labels_ids.shape[0]

    avg_accuracy = total_correct / total_count
    avg_loss = total_eval_loss / len(test_dataloader)

    print(f"Accuracy: {avg_accuracy:.2f}")
    print(f"Validation Loss: {avg_loss:.2f}")


if __name__ == '__main__':
    # In[]: Load model and datasets
    file = os.path.realpath('__file__')
    root = os.path.dirname(file)
    model_name = "sadickam/sdg-classification-bert"
    save_path = os.path.join(root, "SDGclassfierModelConfig")

    if os.path.isdir(save_path):
        model, tokenizer = load_model(model_name, save_path)
    else:
        model, tokenizer = save_model(model_name, save_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_model(model, tokenizer, device)
