import torch


def sampling_class(
    dataloader,
    target_class,
    num_samples,
    num_class,
    positive_sample=True,
    batch_size=16,
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

            while sum(len(ids) for ids in sampled_ids) >= batch_size:
                # Determine the size of the batch to yield
                current_batch_size = 0
                batch_input_ids, batch_masks, batch_labels = [], [], []
                while sampled_ids and current_batch_size < batch_size:
                    ids = sampled_ids[0]
                    masks = sampled_masks[0]
                    labels = sampled_labels[0]

                    # How many samples to take from the current selection
                    take = min(len(ids), batch_size - current_batch_size)
                    batch_input_ids.append(ids[:take])
                    batch_masks.append(masks[:take])
                    batch_labels.append(labels[:take])

                    # Update the current batch size
                    current_batch_size += take

                    # Remove the used samples
                    if take == len(ids):
                        sampled_ids.pop(0)
                        sampled_masks.pop(0)
                        sampled_labels.pop(0)
                    else:
                        sampled_ids[0] = ids[take:]
                        sampled_masks[0] = masks[take:]
                        sampled_labels[0] = labels[take:]

                yield (
                    torch.cat(batch_input_ids),
                    torch.cat(batch_masks),
                    torch.cat(batch_labels),
                    sum(len(ids) for ids in batch_input_ids),
                )

        if all(count >= samples_per_class for count in class_counts.values()):
            break

    if sampled_ids:
        yield (
            torch.cat(sampled_ids),
            torch.cat(sampled_masks),
            torch.cat(sampled_labels),
            sum(len(ids) for ids in sampled_ids),
        )
