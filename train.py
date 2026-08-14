import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from model.transformer import Transformer
import random
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter

SEED = 69

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def log_metrics(log_path, step, train_loss, val_loss, lr):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "train_loss", "val_loss", "lr"])
        writer.writerow([step, train_loss, val_loss, lr])

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_loss(model, idx, criterion):
    inputs = idx[:,:-1]
    targets = idx[:,1:]
    logits, _ = model(inputs)
    batch_size, time_steps, vocab_size = logits.shape
    logits = logits.reshape(batch_size*time_steps, vocab_size)
    targets = targets.reshape(batch_size*time_steps)
    return criterion(logits, targets)

def compute_val_loss(model, val_iter, criterion, eval_iters):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            idx = next(val_iter)
            total_loss += compute_loss(model, idx, criterion).item()
    model.train()
    return total_loss / eval_iters


def get_lr(step, warmup_steps, max_steps, lr_max=3e-4, lr_min=3e-5):
    if step <= warmup_steps:
        return step * (lr_max / warmup_steps)
    progress = (step - warmup_steps) / (max_steps - warmup_steps)  # 0 → 1
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

def stream_tokens(data_path, tokenizer, text_fraction):
    data_size = data_path.stat().st_size
    sample_size = int(data_size * text_fraction)
    chunk_size = 1024 * 1024  # 1MB
    bytes_read = 0
    with data_path.open("r", encoding="utf-8", errors="ignore") as f:
        while bytes_read < sample_size:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk.encode("utf-8"))
            yield tokenizer.encode(chunk).ids

def get_data_iter(data_path, tokenizer, text_fraction, batch_size, context_length, device):
    while True:
        buffer = []
        for ids in stream_tokens(data_path, tokenizer, text_fraction):
            buffer.extend(ids)
            while len(buffer) >= (context_length + 1) * batch_size:
                batch = torch.tensor(
                    [buffer[i:i+context_length+1] for i in range(0, batch_size*(context_length+1), context_length+1)],
                    dtype=torch.long
                ).to(device)
                yield batch
                buffer = buffer[batch_size*(context_length+1):]

def main():

    # 1. config
    DATA_PATH       = Path("data/train.txt")
    VAL_PATH        = Path("data/val.txt")
    TOKENIZER_PATH  = Path("tokenizer.json")
    CHECKPOINT_PATH = Path("param/best_param_final.pt")
    LOG_PATH        = Path("logs/metrics_final.csv") # run A = 1:1, run B = 2:1, run C = transfo pur
    device          = get_device()

    writer = SummaryWriter(log_dir="runs/run_final")

    d_model         = 512
    d_ff            = 4*d_model
    n_layers        = 8
    n_head          = 16
    n_kv_head       = 2
    n_head_memory   = n_head
    n_head_query    = n_head 
    spectral_sample = 2 
    head_dim        = d_model//n_head 
    context_length  = 256
    batch_size      = 16
    max_steps       = 20000
    eval_interval   = 500
    eval_iters      = 100
    log_interval    = 50
    lr_max          = 3e-4
    warmup_steps    = 1000
    grad_clip       = 1.0
    text_fraction   = 0.5
    best_val_loss   = math.inf
    sca_ratio       = 1
    
    # 2. data
    tokenizer  = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    train_iter = get_data_iter(DATA_PATH, tokenizer, text_fraction, batch_size, context_length, device)
    val_iter   = get_data_iter(VAL_PATH,  tokenizer, text_fraction, batch_size, context_length, device)
    
    # 3. model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_layers=n_layers,
        context_length=context_length,
        n_head_memory=n_head_memory,
        n_head_query=n_head_query,
        spectral_sample=spectral_sample,
        head_dim=head_dim,
        sca_ratio= sca_ratio
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    
    print(
        f"device={device} | vocab_size={vocab_size} | text_fraction={text_fraction}"
    )
    print(
        f"config | d_model={d_model} n_head={n_head} n_layers={n_layers} "
        f"context_length={context_length} batch_size={batch_size}"
    )
    
    # 4. training loop
    model.train()
    for step in range(max_steps):
        idx = next(train_iter)
        loss = compute_loss(model, idx, criterion)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        lr = get_lr(step, warmup_steps, max_steps, lr_max=lr_max)    
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        optimizer.step()
        if (step + 1) % log_interval == 0:
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", lr, step)

        should_eval = (step == 0) or ((step + 1) % eval_interval == 0)

        if should_eval:
            val_loss = compute_val_loss(model, val_iter, criterion, eval_iters)
            writer.add_scalar("loss/val", val_loss, step)
            log_metrics(LOG_PATH, step+1, loss.item(), val_loss, lr)
        
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
                            "n_kv_head": n_kv_head, 
                            "n_layers": n_layers,
                            "sca_ratio": sca_ratio,
                            "context_length": context_length,
                            "n_head_memory": n_head_memory,
                            "n_head_query": n_head_query, 
                            "spectral_sample": spectral_sample,
                            "head_dim": head_dim, 
                            "best_val_loss": best_val_loss,
                            "step": step,
                            "tokenizer_path": str(TOKENIZER_PATH),
                        },
                        CHECKPOINT_PATH,)
                print(f"checkpoint saved | path={CHECKPOINT_PATH} | best_val_loss={best_val_loss:.4f}")
        
        
if __name__ == "__main__":
    main()
