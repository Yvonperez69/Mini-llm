import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer

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


