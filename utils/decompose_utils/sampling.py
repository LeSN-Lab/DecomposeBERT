import torch


def sampling_class(
    dataloader,
    target_class,
    num_samples,
    num_class,
    positive_sample=True,
    device="cuda",
):
    sampled_ids = []
    sampled_masks = []
    sampled_labels = []
    class_counts = {i: 0 for i in range(num_class)}
    total_sampled = 0

    if positive_sample:
        samples_per_class = num_samples
    else:
        samples_per_class = num_samples // (num_class - 1)

    for batch in dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_masks = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        for class_id in class_counts.keys():
            if class_counts[class_id] >= samples_per_class:
                continue

            if positive_sample and class_id != target_class:
                continue

            mask = b_labels == class_id
            selected_input_ids = b_input_ids[mask]
            selected_attention_masks = b_attention_masks[mask]
            selected_labels = b_labels[mask]

            num_selected = selected_labels.size(0)
            selected = num_selected + class_counts[class_id]
            if selected > samples_per_class:
                num_selected = samples_per_class - class_counts[class_id]
                selected_input_ids = selected_input_ids[:num_selected]
                selected_attention_masks = selected_attention_masks[:num_selected]
                selected_labels = selected_labels[:num_selected]

            class_counts[class_id] += num_selected
            total_sampled += num_selected

            sampled_ids.append(selected_input_ids)
            sampled_masks.append(selected_attention_masks)
            sampled_labels.append(selected_labels)

        if all(count >= samples_per_class for count in class_counts.values()):
            break

    sampled_ids = torch.cat(sampled_ids)
    sampled_masks = torch.cat(sampled_masks)
    sampled_labels = torch.cat(sampled_labels)

    return sampled_ids, sampled_masks, sampled_labels, total_sampled
