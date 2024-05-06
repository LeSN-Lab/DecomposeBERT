from transformers import AutoTokenizer


def save_tokenizer(model_config):
    model_name = model_config.model_name
    model_dir = model_config.config_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)

    return tokenizer


def load_tokenizer(model_config):
    config_path = model_config.config_dir
    if not model_config.is_downloaded:
        tokenizer = save_tokenizer(model_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_path)
    return tokenizer