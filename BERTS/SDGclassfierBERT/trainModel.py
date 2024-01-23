# In[]
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.modelConfig import save_model, load_model
from utils.load_dataset import load_sgd
import torch
from tqdm import tqdm


# In[]: Load model and datasets
file = os.path.realpath(__file__)
root = os.path.dirname(file)
model_name = "sadickam/sdg-classification-bert"
save_path = os.path.join(root, "SDGclassfierModelConfig")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if os.path.isdir(save_path):
    model, tokenizer = load_model(model_name, save_path)
else:
    model, tokenizer = save_model(model_name, save_path)
model = model.to(device)

# In[]: Train model
def train_model(model, tokenizer):
    train_dataloader, validation_dataloader = load_sgd(tokenizer)

    # In[] : Set optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3

    # In[] : Scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # In[]: Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, disable=False)
        for batch in progress_bar:
            # Load batch data in GPU
            batch = tuple(t.to(device) for t in batch)
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
            total_loss += loss.item()
    test_model(model, tokenizer)

def test_model(model, tokenizer):
    train_dataloader, validation_dataloader = load_sgd(tokenizer)

    model.eval()

    # Initialize accuracy variables
    total_eval_accuracy = 0
    total_eval_loss = 0

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(validation_dataloader, desc="Evaluating"):
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]

        loss = loss_fn(logits, b_labels)
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, labels_ids)
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)


if __name__ == '__main__':
    train_model(model, tokenizer)