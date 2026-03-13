import torch
from tokenizers import Tokenizer
from model.transformer import Transformer

temperature = 0.8
top_k = 40
repetition_penalty = 1.2

tokenizer = Tokenizer.from_file('tokenizer.json')

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

path = '/Users/yvonperez/Dropbox/Mac/Documents/Info/Attention/param/best_param.pt'
checkpoint = torch.load(path, map_location=device)

vocab_size = checkpoint['vocab_size']
d_model = checkpoint['d_model']
d_ff = 4*d_model
n_head = checkpoint['n_head']
n_layers = checkpoint['n_layers']
context_length = checkpoint['context_length']
max_new_tokens = 35
top_k = max(1, min(top_k, vocab_size))
eos_token_id = tokenizer.token_to_id("<eos>")

model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_head=n_head, n_layers=n_layers).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prompt = input()


ids = torch.tensor(tokenizer.encode(prompt).ids).to(torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    for _ in range(max_new_tokens):
        inputs = ids[:, -context_length:]
        logits = model(inputs)
        
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        for token_id in ids[0].tolist():
            next_token_logits[:, token_id] /= repetition_penalty

        topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = torch.gather(topk_indices, 1, next_token)
        
        ids = torch.cat([ids, next_token_id], dim=1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    out = tokenizer.decode(ids.squeeze(0).tolist())

    print(out)
