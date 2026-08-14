import torch
from tokenizers import Tokenizer
from model.transformer import Transformer
import sys
sys.stdout.reconfigure(encoding='utf-8')

temperature = 0.4
top_k = 40
repetition_penalty = 1.3

tokenizer = Tokenizer.from_file('tokenizer.json')

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

path = 'param/best_param.pt'
checkpoint = torch.load(path, map_location=device, weights_only=True)

vocab_size     = checkpoint['vocab_size']
d_model        = checkpoint['d_model']
d_ff           = checkpoint.get('d_ff', 4 * d_model)
n_head         = checkpoint['n_head']
n_kv_head      = checkpoint.get('n_kv_head', 2)
n_layers       = checkpoint['n_layers']
context_length = checkpoint['context_length']
sca_ratio      = checkpoint.get('sca_ratio', 1)
n_head_memory  = checkpoint.get('n_head_memory', n_head)
n_head_query   = checkpoint.get('n_head_query', n_head)
spectral_sample = checkpoint.get('spectral_sample', 2)
head_dim       = checkpoint.get('head_dim', d_model // n_head)
max_new_tokens = 35
top_k = max(1, min(top_k, vocab_size))
eos_token_id = tokenizer.token_to_id("<eos>")

model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    n_head=n_head,
    n_kv_head=n_kv_head,
    n_layers=n_layers,
    context_length=context_length,
    sca_ratio=sca_ratio,
    n_head_memory=n_head_memory,
    n_head_query=n_head_query,
    spectral_sample=spectral_sample,
    head_dim=head_dim,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('prompt : ')
prompt = input()
print('réponse : ')


ids = torch.tensor(tokenizer.encode(prompt).ids).to(torch.long).unsqueeze(0).to(device)
states = None

with torch.no_grad():
    logits, states = model(ids, states= states)

    for _ in range(max_new_tokens):
        # génération récurrente - un token à la fois
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

        for token_id in set(ids[0].tolist()) :
            if next_token_logits[:, token_id] <=0.0 : 
                next_token_logits[:, token_id] *= repetition_penalty
            else : 
                next_token_logits[:, token_id] /= repetition_penalty 

        topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_id = torch.gather(topk_indices, 1, next_token)
        ids = torch.cat([ids, next_token_id], dim=1)

        logits, states = model(next_token_id, states) 

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    out = tokenizer.decode(ids.squeeze(0).tolist())

print(out)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(out)     
print("output sauvegardé dans output.txt")
