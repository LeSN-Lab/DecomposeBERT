# In[]: Import Libraries
import torch
from utils.model_utils.train_model import train_model
from utils.model_utils.evaluate import evaluate_model
from utils.model_utils.load_model import load_classification_model
from utils.dataset_utils.load_dataset import load_data
from utils.model_utils.model_config import ModelConfig
from transformers import AutoConfig, TrainingArguments
from utils.model_utils.load_tokenizer import load_tokenizer


# In[]: Train model Examples
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    model_type = "bert"
    data = "OSDG"
    num_labels = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_name = None
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model_config = ModelConfig(
        _model_name=model_name,
        _model_type=model_type,
        _data=data,
        _transformer_config=config,
        _checkpoint_name=checkpoint_name,
        _device=device,
    )

    # If you have a checkpoint, uncomment this
    """
    model_config.model_config.checkpoint_name = 'epoch_1.pt'
    """

    # Train model

    model, tokenizer, checkpoint = load_classification_model(
        model_config, train_mode=True
    )
    train_dataset, valid_dataset, test_dataset = load_data(
        model_config=model_config,
        batch_size=16,
        test_size=0.2,
    )
    training_args = TrainingArguments(
        output_dir=model_config.train_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=model_config.train_dir,
        warmup_steps=1000,
        weight_decay=0.01,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        learning_rate=5e-5,
        max_steps=3000,
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    trainer = train_model(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )

    # If you want to evaluate an accuracy of the model, uncomment this
    # Evaluate model
    result = trainer.evaluate()
    print(result)
