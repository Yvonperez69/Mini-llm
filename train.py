import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from model.transformer import Transformer
import random
import numpy as np

SEED = 69

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def read_text_sample(data_path, text_fraction):
    data_size = data_path.stat().st_size
    sample_size = max(1, int(data_size*text_fraction))
    
    with data_path.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(sample_size)
    return text, sample_size, data_size

def get_batch(tokens, batch_size, context_length, device):
    start_idx = torch.randint(0, len(tokens)-context_length,(batch_size,))
    batch = torch.stack([tokens[start : start + context_length + 1] for start in start_idx])
    return batch.to(device)

def compute_loss(model, idx, criterion):
    inputs = idx[:,:-1]
    targets = idx[:,1:]
    logits = model(inputs)
    batch_size, time_steps, vocab_size = logits.shape
    logits = logits.reshape(batch_size*time_steps, vocab_size)
    targets = targets.reshape(batch_size*time_steps)
    return criterion(logits, targets)

def compute_val_loss(model, val_tokens, batch_size, context_length, criterion, device, eval_iters):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            idx = get_batch(val_tokens, batch_size, context_length, device)
            total_loss += compute_loss(model, idx, criterion).item()

    model.train()
    return total_loss / eval_iters

"""
scheduler cosine à implémenter Jeudi
def get_lr():
    pass

"""
def main():
    
    # 1. config
    DATA_PATH       = Path("data/input.txt")
    TOKENIZER_PATH  = Path("tokenizer.json")
    CHECKPOINT_PATH = Path("param/best_param.pt")
    device          = torch.device("mps")  # M1

    d_model         = 512
    d_ff            = 4*d_model
    n_layers        = 10
    n_head          = 8
    n_kv_head       = 2
    context_length  = 256
    batch_size      = 16
    max_steps       = 30000
    eval_interval   = 500
    eval_iters      = 100
    log_interval    = 50
    lr_max          = 3e-4
    warmup_steps    = 500
    grad_clip       = 1.0
    text_fraction   = 0.05
    best_val_loss   = math.inf
    
    # 2. data
    text, sample_size, data_size = read_text_sample(DATA_PATH, text_fraction)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    encoding = tokenizer.encode(text)
    tokens = torch.tensor(encoding.ids, dtype=torch.long)
    vocab_size = tokenizer.get_vocab_size()
    
    train_split = int(0.9*len(tokens))
    train_tokens = tokens[:train_split]
    val_tokens = tokens[train_split:]
    
    # 3. model
    model = Transformer(vocab_size=vocab_size,
                        d_model=d_model,
                        d_ff=d_ff,
                        n_head=n_head,
                        n_layers=n_layers
                        ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    
    print(
        f"device={device} | sample={sample_size / 1024 / 1024:.1f}MB/{data_size / 1024 / 1024:.1f}MB "
        f"| vocab_size={vocab_size} | train_tokens={len(train_tokens)} | val_tokens={len(val_tokens)}"
    )
    print(
        f"config | d_model={d_model} n_head={n_head} n_layers={n_layers} "
        f"context_length={context_length} batch_size={batch_size}"
    )
    
    # 4. training loop
    model.train()
    for step in range(max_steps):
        idx = get_batch(train_tokens, batch_size, context_length, device)
        loss = compute_loss(model, idx, criterion)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if (step + 1) % log_interval == 0:
            print(f"step : {step + 1}/{max_steps} | train_loss = {loss:.4f}")
        should_eval = (step == 0) or ((step + 1) % eval_interval == 0)
        if should_eval:
            val_loss = compute_val_loss(model, val_tokens, batch_size, context_length, criterion, device, eval_iters)
            print(f"step {step + 1}/{max_steps} | val_loss={val_loss:.4f}")
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(                    {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "vocab_size": vocab_size,
                            "d_model": d_model,
                            "d_ff": d_ff,
                            "n_head": n_head,
                            "n_layers": n_layers,
                            "context_length": context_length,
                            "best_val_loss": best_val_loss,
                            "step": step,
                            "tokenizer_path": str(TOKENIZER_PATH),
                        },
                        CHECKPOINT_PATH,)
                print(f"checkpoint saved | path={CHECKPOINT_PATH} | best_val_loss={best_val_loss:.4f}")
        
        
if __name__ == "__main__":
    main()