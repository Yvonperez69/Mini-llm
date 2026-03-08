import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
import matplotlib.pyplot as plt


vocab_size = 5000
d_model = 256
d_ff = 4*d_model
n_head = 8
n_layers = 6
batch_size = 8
context_length = 128
lr = 5e-4
max_steps = 10000

device = torch.device("cuda"  if torch.cuda.is_available() else "mps")


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

loss_list = []

# boucle d'entrainement
if train :
    model = model.train()

idx = torch.randint(0, vocab_size, (batch_size,context_length+1)).to(device=device)

for step in range(max_steps):
    loss = training_step(model=model, optimizer=optimizer, idx=idx, criterion=criterion)
    loss_list.append(loss)
    

plt.plot([step for step in range(max_steps)], loss_list)
plt.show()