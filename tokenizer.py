from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
def tokenize_batch(batch):
    en_texts = [example["en"] for example in batch["translation"]] #batch is a dictionar containing translation pairs, extracting english pairs from batch[translation]
    ur_texts = [example["ur"] for example in batch["translation"]] #batch["translation"] is assumed to be a list of dictionaries, where each dictionary contains a translation pair (e.g., {"en": "Hello", "ur": "ہیلو"}).

    tokenized_inputs = tokenizer(en_texts, padding="max_length", truncation=True)
    tokenized_labels = tokenizer(ur_texts, padding="max_length", truncation=True)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_labels["input_ids"]
    }

tokenized_dataset = filtered_dataset.map(tokenize_batch, batched=True)
def tokenize_batch(batch):
    en_texts = [example["en"] for example in batch["translation"]]
    ur_texts = [example["ur"] for example in batch["translation"]]

    tokenized_inputs = tokenizer(en_texts, padding="max_length", truncation=True)
    tokenized_labels = tokenizer(ur_texts, padding="max_length", truncation=True)

    print("Example English Text:", en_texts[0])
    print("Tokenized Input IDs:", tokenized_inputs["input_ids"][0])

    print("Example Urdu Text:", ur_texts[0])
    print("Tokenized Label IDs:", tokenized_labels["input_ids"][0])  # Check labels before returning

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_labels["input_ids"]
    }
example_batch = dataset["train"][:5]  # Get a small batch for demonstration
tokenized_data = tokenize_batch(example_batch)
tokenized_labels = tokenized_data # Extract tokenized_labels from the output

print(tokenized_labels["input_ids"][:3])  
print(tokenizer.pad_token_id)
