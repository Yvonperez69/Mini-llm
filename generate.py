import torch
from tokenizer import Tokenizer
from model.transformer import Transformer

tokenizer = Tokenizer()

device = torch.device("cuda"  if torch.cuda.is_available() else "mps")

path = '/Users/yvonperez/Dropbox/Mac/Documents/Info/Attention/param/best_param.pt'
checkpoint = torch.load(path)

word2id = checkpoint['vocab']
id2word = tokenizer.decode(word2id)

vocab_size = checkpoint['vocab_size']
d_model = checkpoint['d_model']
d_ff = 4*d_model
n_head = checkpoint['n_head']
n_layers = checkpoint['n_layers']

model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_head=n_head, n_layers=n_layers).to(device)

