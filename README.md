Structure du projet

/model :
    attention.py
    feedforward.py
    block.py
    transformer.py

/tokenizer

train.py
generate.py
config.py


Hyperparamètres raisonnables (trainable sur machine perso)
vocab_size ≈ 8k–16k
context_length = 256
d_model = 256
n_heads = 8
n_layers = 6
d_ff = 4 × d_model
dropout = 0.1



La suite pour entraîner ton mini-LLM, c’est surtout de compléter le pipeline autour du modèle (données + boucle d’entraînement), mais avant ça tu as quelques bugs bloquants à corriger.

**Blocages à corriger avant entraînement**
- `train.py` : import probablement faux (`from transformer import Transformer`) si ton fichier est dans `model/transformer.py` -> plutôt `from model.transformer import Transformer`.
- `model/transformer.py` : `math.sin` / `math.cos` sur des tenseurs -> il faut `torch.sin` / `torch.cos`.
- `model/transformer.py`, `model/block.py` : imports relatifs à corriger (`from .block import ...`, `from .attention import ...`, `from .feedforward import ...`) si tu exécutes depuis la racine.
- `model/attention.py` : bug de reshape, tu réutilises `q` pour `k` et `v`.
  - `v = v.view(...)` (pas `q.view`)
  - `k = k.view(...)` (pas `q.view`)
- `model/attention.py` : `attention.masked_fill(...)` n’est pas réassigné -> le masque n’est pas appliqué.
- `model/feedforward.py` : `nn.Linear(d_model, d_ff)` et `nn.Linear(d_ff, d_model)` plantent si `d_ff=None`; utilise `self.d_ff`.

**Ce qu’il manque pour vraiment entraîner**
1. Tokenizer + vocab (`/tokenizer`)  
2. Chargement du corpus texte  
3. Encodage en IDs + split `train/val`  
4. Batching en fenêtres `(B, T)` (random chunks)  
5. Boucle d’entraînement complète (epochs/steps, logs)
6. Validation périodique (`model.eval()`, loss val)
7. Checkpoints (`torch.save`)
8. Génération de test pour vérifier la qualité

**Pipeline minimal (ordre recommandé)**
1. Corriger les bugs ci-dessus.
2. Créer un `get_batch(split)` qui renvoie un batch `idx` de shape `(B, T+1)` (tu utilises déjà `inputs=idx[:,:-1]`, `targets=idx[:,1:]`).
3. Instancier :
   - `model = Transformer(...)`
   - `optimizer = torch.optim.AdamW(...)`
   - `criterion = nn.CrossEntropyLoss()`
4. Boucle :
   - `for step in range(max_steps):`
   - `loss = training_step(...)`
   - tous les `eval_interval`: loss train/val + save checkpoint
5. Tester `generate.py` ensuite.

**Hyperparams de départ (machine perso)**
- `context_length=128` (plus safe que 256 au début)
- `batch_size=16` (à ajuster selon VRAM)
- `d_model=256`, `n_heads=8`, `n_layers=4-6`
- `lr=3e-4`
- `weight_decay=0.1`
- `dropout=0.1`
- `grad_clip=1.0`

**Important**
Ton `train.py` actuel contient seulement une `training_step`, donc tu n’as pas encore un script d’entraînement complet.

Si tu veux, je peux te faire la suite directement dans le code :
1. corriger les bugs du modèle
2. te générer un `train.py` minimal complet (avec `get_batch`, eval, checkpoint)
3. ajouter un `generate.py` basique pour tester après entraînement