# Loss function & Optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)

PAD_TOKEN = 0

def generate_src_mask(attention_mask):
    """Creates a mask for the source sequence using tokenizer's attention_mask."""
    src_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()  # [batch_size, 1, 1, seq_len]
    src_mask = src_mask.expand(-1, 1, attention_mask.size(1), -1)  # Ensure compatibility with attention
    src_mask = (1.0 - src_mask) * -1e9  # Convert 1s/0s into large negative values
    return src_mask


def generate_tgt_mask(tgt, attention_mask):
    """Creates a target mask combining padding and causal masks."""
    batch_size, seq_len = tgt.shape  #  Extract sequence length from `tgt`
   

    # Ensure `attention_mask` has the same `seq_len`
    if attention_mask.shape[1] != seq_len:
        raise ValueError(f"Mismatch: tgt seq_len={seq_len}, attention_mask seq_len={attention_mask.shape[1]}")

    # Padding mask: (batch_size, 1, 1, seq_len)
    padding_mask = attention_mask[:, :seq_len].unsqueeze(1).unsqueeze(2).float()
    padding_mask = (1.0 - padding_mask) * -1e9  # Convert 0s to large negative values
    padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len) 

    # Causal mask: (1, 1, seq_len, seq_len)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, seq_len, seq_len)

    # Combine both masks
    tgt_mask = padding_mask + causal_mask  
   
    return tgt_mask
def train():
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    for batch in train_dataloader:
        src = torch.stack([torch.as_tensor(seq, dtype=torch.long).clone().detach() for seq in batch["input_ids"]]).to(DEVICE)
        tgt = torch.stack([torch.as_tensor(seq, dtype=torch.long).clone().detach() for seq in batch["labels"]]).to(DEVICE)

        # Generate masks, converting to PyTorch tensors first
        src_attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)  # Convert to tensor
        tgt_attention_mask = torch.tensor(batch["attention_mask"]).to(DEVICE)  # Convert to tensor

        # Generate masks using correct inputs
        src_mask = generate_src_mask(src_attention_mask)
        tgt_mask = generate_tgt_mask(tgt[:, :-1], tgt_attention_mask)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)

        # Compute loss, ignoring PAD tokens
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        total_loss += loss.item()

        # Compute token accuracy, ignoring PAD tokens
        preds = output.argmax(dim=-1)
        non_pad_mask = tgt[:, 1:] != tokenizer.pad_token_id
        correct_tokens += ((preds == tgt[:, 1:]) & non_pad_mask).sum().item()
        total_tokens += non_pad_mask.sum().item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

    # Compute averages
    avg_loss = total_loss / len(train_dataloader)
    token_acc = correct_tokens / total_tokens

    return avg_loss, token_acc
EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss, train_acc = train()
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")
