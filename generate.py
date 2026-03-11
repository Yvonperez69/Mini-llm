import torch
from tokenizer import Tokenizer
from model.transformer import Transformer

tokenizer = Tokenizer()

device = torch.device("cuda"  if torch.cuda.is_available() else "mps")

path = '/Users/yvonperez/Dropbox/Mac/Documents/Info/Attention/param/best_param.pt'
checkpoint = torch.load(path)

tokenizer.word2id = checkpoint['vocab']
tokenizer.id2word = {i:w for w, i in tokenizer.word2id.items()}

vocab_size = checkpoint['vocab_size']
d_model = checkpoint['d_model']
d_ff = 4*d_model
n_head = checkpoint['n_head']
n_layers = checkpoint['n_layers']
context_length = checkpoint['context_length']
max_new_tokens = 20

model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_head=n_head, n_layers=n_layers).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prompt = "What is your name"


ids = torch.Tensor(tokenizer.encode(prompt)).to(torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    for _ in range(max_new_tokens):
        inputs = ids[:, -context_length:]
        logits = model(inputs)
        next_token_id = torch.argmax(logits[:,-1,:], dim=1,keepdim=True)
        
        ids = torch.cat([ids, next_token_id], dim=1)

    out = tokenizer.decode(ids.squeeze(0).tolist())

    print(out)