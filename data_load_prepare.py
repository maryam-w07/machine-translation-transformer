
from datasets import load_dataset

# Load the OPUS100 dataset for English-Urdu
dataset = load_dataset("opus100", "en-ur")

# Inspect the dataset
print(dataset["train"][0])
def filter_empty_translations(batch):
    filtered_batch = {
        "translation": [
            entry for entry in batch["translation"]
            if entry["en"] and entry["ur"]
        ]
    }
    return filtered_batch

filtered_dataset = dataset.map(filter_empty_translations, batched=True)

import torch
BATCH_SIZE = 4  
EPOCHS = 5
CLIP_NORM = 1.0  # Gradient clipping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloaders
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=BATCH_SIZE)
