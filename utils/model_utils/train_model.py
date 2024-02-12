# In[]: Import Libraries
import os
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.model_utils.load_model import load_classification_model
from utils.model_utils.evaluate import evaluate_model
from utils.data_utils.load_dataset import load_sdg
import torch
from tqdm import tqdm
from utils.paths import p


# In[]: Train model
def train_model(
    model_config,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    test=True,
):
    train_path = model_config.train_dir
    model, tokenizer, checkpoint = load_classification_model(model_config)

    # In[] : Load model
    train_dataloader, valid_dataloader, test_dataloader = load_sdg(
        model_config, tokenizer, batch_size=batch_size, test_size=0.2
    )

    # In[] : Set optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    start_epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")

    best_val_loss = np.inf
    no_improve_epochs = 0
    early_stopping_threshold = 3

    # In[]: Training loop
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
                disable=False,
            )
            for batch in progress_bar:
                # Load batch data in GPU

                b_input_ids = batch["input_ids"].to(model_config.device)
                b_attention_masks = batch["attention_mask"].to(model_config.device)
                b_labels = batch["labels"].to(model_config.device)

                # Initialize gradient to 0
                model.zero_grad()

                # Forward pass
                outputs = model(
                    b_input_ids, attention_mask=b_attention_masks, labels=b_labels
                )
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
                avg_loss = total_loss / (len(train_dataloader))
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
                )

            # In[]: Validation
            val_accuracy, val_loss = evaluate_model(model_config, model, valid_dataloader)
            print(
                f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epochs = epoch + 1

                # Save model
                best_model_path = os.path.join(train_path, f"epoch_{best_epochs}.pt")
                torch.save(
                    {
                        "epoch": best_epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    best_model_path,
                )

                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_threshold:
                    print("Early stopping triggered.")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                        },
                        os.path.join(train_path, "best_model.pt"),
                    )
                    break
        best_model_path = os.path.join(train_path, "best_model.pt")
        if test and os.path.isfile(best_model_path):
            print(f"Loading best model for testing")
            checkpoint_name = "best_model.pt"
            model_config.checkpoint_name = checkpoint_name
            model, _, _ = load_classification_model(model_config)
            evaluate_model(model, model_config, test_dataloader)

    except Exception as e:
        print(f"An error occurred: {e}")
    return model
