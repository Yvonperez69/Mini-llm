import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
import math
from pathlib import Path

DATA_PATH = Path("/Users/yvonperez/Dropbox/Mac/Documents/Info/Attention/data/input.txt")
TEXT_FRACTION = 0.01

with open(DATA_PATH, "rb") as f:
    text_bytes = f.read(max(1, int(DATA_PATH.stat().st_size * TEXT_FRACTION)))

text = text_bytes.decode("utf-8", errors="ignore")

d_model = 256
d_ff = 4*d_model
n_head = 8
n_layers = 6
batch_size = 8
context_length = 128
lr = 3e-4
max_steps = 30000

device = torch.device("cuda"  if torch.cuda.is_available() else "mps")


tokenizer = Tokenizer.from_file("tokenizer.json")
encoding = tokenizer.encode(text)
tokens = torch.tensor(encoding.ids).to(torch.long)
vocab_size = tokenizer.get_vocab_size()


model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_head=n_head, n_layers=n_layers).to(device)

train = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)



def training_step(model, optimizer, idx, criterion):
    # idx : (B ,T)
    # input et target
    inputs = idx[:, :-1]
    targets = idx[:, 1:]

    # forward 

    logits = model(inputs) # (B,T-1,vocab_size)
    B, T, V = logits.shape

    # reshape pour calculer la loss
    logits = logits.reshape(B*T,V)
    targets = targets.reshape(B*T)

    # loss
    loss = criterion(logits, targets)

    # Backpropagation
    optimizer.zero_grad() # on remet les gradients à zéro avant de faire la backpropagation
    loss.backward() # calcul des gradients
    optimizer.step() # mise à jour des poids

    return loss.item() 

def estimate_val_loss(model, val_tokens, batch_size, context_length, criterion, device, eval_iters):
    model = model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for _ in range(eval_iters):
            idx = get_batch(val_tokens, batch_size, context_length, device)
            inputs = idx[:, :-1]
            targets = idx[:, 1:]
            
            # forward 

            logits = model(inputs) # (B,T-1,vocab_size)
            B, T, V = logits.shape

            # reshape pour calculer la loss
            logits = logits.reshape(B*T,V)
            targets = targets.reshape(B*T)            
            
            # loss
            loss = criterion(logits, targets)
            total_loss += loss
    
    model = model.train()
            
    return (total_loss/eval_iters).item()
        
     

def get_batch(tokens, batch_size, context_length, device):

    start_indices = torch.randint(0, len(tokens) - context_length, (batch_size,))

    batch = torch.stack([tokens[start : start + context_length+1] for start in start_indices])

    batch = batch.to(device)

    return batch
loss_list = []
val_loss_list = []
best_val_loss = math.inf

# split train/val
split_idx = int(0.9*len(tokens))
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

N = 200

# boucle d'entrainement
if train :
    model = model.train()
    
    for step in range(max_steps):
        
        idx = get_batch(train_tokens, batch_size, context_length, device)
        loss = training_step(model=model, optimizer=optimizer, idx=idx, criterion=criterion)
        loss_list.append(loss)
        
        if step % N == 0:
            val_loss = estimate_val_loss(model, val_tokens, batch_size, context_length, criterion, device, eval_iters=N)
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                
                # save checkpoint
                checkpoint = {
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'vocab_size': vocab_size,
                    'd_model': d_model,
                    'n_head': n_head,
                    'n_layers': n_layers,
                    'context_length': context_length,
                    'best_val_loss' : best_val_loss,
                    'step' : step
                }
                torch.save(checkpoint,'/Users/yvonperez/Dropbox/Mac/Documents/Info/Attention/param/best_param.pt')
                print(f"checkpoint saved | best_val_loss={best_val_loss:.4f}")
            
            val_loss_list.append(val_loss)
            print(f"step {step}/{max_steps} | val_loss={val_loss:.4f}")

        
        print(f"step {step}/{max_steps} | train_loss={loss:.4f}")

        
    plt.plot([step for step in range(int(max_steps/N))], val_loss_list)
    plt.plot([step for step in range(max_steps)], loss_list)
    plt.show()
