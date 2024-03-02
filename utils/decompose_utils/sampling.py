import torch


def sampling_class(
    dataloader,
    target_class,
    num_samples,
    positive_sample=True,
    device="cuda",
):
    sampled_ids = []
    sampled_masks = []
    sampled_labels = []

    for batch in dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_masks = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)
        if positive_sample:
            mask = b_labels == target_class
        else:
            mask = b_labels != target_class

        selected_input_ids = b_input_ids[mask]
        selected_attention_masks = b_attention_masks[mask]
        selected_labels = b_labels[mask]

        sampled_ids.append(selected_input_ids)
        sampled_masks.append(selected_attention_masks)
        sampled_labels.append(selected_labels)

        if sum(len(labels) for labels in sampled_labels) >= num_samples:
            break

    sampled_ids = torch.cat(sampled_ids)[:num_samples]
    sampled_masks = torch.cat(sampled_masks)[:num_samples]
    sampled_labels = torch.cat(sampled_labels)[:num_samples]

    return sampled_ids, sampled_masks, sampled_labels
