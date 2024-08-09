import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, Generator
import random


class SamplingDataset(IterableDataset):
    def __init__(
            self,
            dataloader: DataLoader,
            target_class: int,
            num_samples: int,
            num_class: int,
            positive_sample: bool = True,
            batch_size: int = 4,
            device: torch.device = torch.device("cuda:0"),
            seed: int = 42,
            resample: bool = True,
    ) -> None:
        if num_samples % batch_size != 0:
            raise ValueError("num_samples must be divisible by batch_size")

        self.dataloader = dataloader
        self.target_class = target_class
        self.num_samples = num_samples
        self.num_class = num_class
        self.batch_size = batch_size
        self.positive_sample = positive_sample
        self.device = device
        self.resample = resample

        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.sampled_data = None  # 샘플된 데이터를 저장할 공간

    def _sample_data(self) -> None:
        # 샘플된 데이터를 초기화하여 이전 샘플을 제거
        self.sampled_data = []

        sampled_ids = []
        sampled_masks = []
        sampled_labels = []

        class_sample_counts = {}
        if self.positive_sample:
            class_sample_counts[self.target_class] = self.num_samples
        else:
            available_classes = [
                i for i in range(self.num_class) if i != self.target_class
            ]
            samples_per_class = self.num_samples // len(available_classes)
            remainder = self.num_samples % len(available_classes)

            for class_id in available_classes:
                class_sample_counts[class_id] = samples_per_class + (
                    1 if remainder > 0 else 0
                )
                remainder -= 1

        total_sampled = {class_id: 0 for class_id in class_sample_counts}

        for batch in self.dataloader:
            b_input_ids = batch["input_ids"].to(self.device)
            b_attention_masks = batch["attention_mask"].to(self.device)
            b_labels = batch["labels"].to(self.device)

            for class_id, target_count in class_sample_counts.items():
                if total_sampled[class_id] >= target_count:
                    continue

                mask = b_labels == class_id
                selected_input_ids = b_input_ids[mask]
                selected_attention_masks = b_attention_masks[mask]
                selected_labels = b_labels[mask]

                num_selected = selected_labels.size(0)
                if num_selected == 0:
                    continue

                remaining_samples = target_count - total_sampled[class_id]
                num_selected = min(num_selected, remaining_samples)

                selected_input_ids = selected_input_ids[:num_selected]
                selected_attention_masks = selected_attention_masks[:num_selected]
                selected_labels = selected_labels[:num_selected]

                total_sampled[class_id] += num_selected

                if num_selected > 0:
                    sampled_ids.append(selected_input_ids)
                    sampled_masks.append(selected_attention_masks)
                    sampled_labels.append(selected_labels)

                while sum(len(ids) for ids in sampled_ids) >= self.batch_size:
                    current_batch_size = 0
                    batch_input_ids, batch_masks, batch_labels = [], [], []
                    while sampled_ids and current_batch_size < self.batch_size:
                        ids = sampled_ids[0]
                        masks = sampled_masks[0]
                        labels = sampled_labels[0]

                        take = min(len(ids), self.batch_size - current_batch_size)
                        batch_input_ids.append(ids[:take])
                        batch_masks.append(masks[:take])
                        batch_labels.append(labels[:take])

                        current_batch_size += take

                        if take == len(ids):
                            sampled_ids.pop(0)
                            sampled_masks.pop(0)
                            sampled_labels.pop(0)
                        else:
                            sampled_ids[0] = ids[take:]
                            sampled_masks[0] = masks[take:]
                            sampled_labels[0] = labels[take:]

                    if batch_input_ids:
                        sampled_batch = {
                            "input_ids": torch.cat(batch_input_ids),
                            "attention_mask": torch.cat(batch_masks),
                            "labels": torch.cat(batch_labels),
                        }
                        self.sampled_data.append(sampled_batch)

        for class_id, target_count in class_sample_counts.items():
            if total_sampled[class_id] < target_count:
                raise ValueError(
                    f"Could only sample {total_sampled[class_id]} out of {target_count} requested samples for class {class_id}."
                )

        if sampled_ids:
            remaining_batch = {
                "input_ids": torch.cat(sampled_ids),
                "attention_mask": torch.cat(sampled_masks),
                "labels": torch.cat(sampled_labels),
            }
            self.sampled_data.append(remaining_batch)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        if self.sampled_data is None or self.resample:
            self._sample_data()
        for batch in self.sampled_data:
            yield batch
