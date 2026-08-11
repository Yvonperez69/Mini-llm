# Attention

Mini-LLM causal en PyTorch, entraine sur un corpus francophone et organise comme une pipeline locale simple:

1. extraire un corpus texte depuis Wikipedia
2. entrainer un tokenizer BPE
3. entrainer un Transformer causal
4. generer du texte depuis un checkpoint

Le projet reste compact et pedagogique, mais les scripts sont maintenant coherents entre eux: memes chemins par defaut, CLI explicite, checkpoint relancable, et selection de device propre.

## Structure

```text
.
├── model/
│   ├── attention.py
│   ├── block.py
│   ├── feedforward.py
│   └── transformer.py
├── project_utils.py
├── extract_wiki_txt.py
├── tokenizer.py
├── train.py
├── generate.py
└── tokenizer.json
```

## Installation

```bash
pip install torch tokenizers mwparserfromhell
```

## Pipeline

### 1. Extraire le corpus

Place un dump Wikipedia dans le repo, par exemple `frwiki-latest-pages-articles.xml.bz2`, puis lance:

```bash
python extract_wiki_txt.py
```

Par defaut, le script lit `frwiki-latest-pages-articles.xml.bz2` et ecrit le corpus nettoye dans `data/input.txt`.

Options utiles:

```bash
python extract_wiki_txt.py --input frwiki-latest-pages-articles.xml.bz2 --output data/input.txt --min-chars 200
```

### 2. Entrainer le tokenizer

```bash
python tokenizer.py
```

Options utiles:

```bash
python tokenizer.py --input data/input.txt --output tokenizer.json --vocab-size 16000 --min-frequency 2
```

### 3. Entrainer le modele

```bash
python train.py
```

Par defaut, l'entrainement:

- charge `1%` du corpus en memoire
- utilise `tokenizer.json`
- sauvegarde le meilleur checkpoint dans `param/best_param.pt`
- choisit automatiquement `cuda`, puis `mps`, puis `cpu`

Exemple plus explicite:

```bash
python train.py --fraction 0.02 --context-length 256 --batch-size 8 --max-steps 5000
```

Reprendre un entrainement existant:

```bash
python train.py --resume --max-steps 2000
```

### 4. Generer du texte

```bash
python generate.py --prompt "La France est"
```

Ou en interactif:

```bash
python generate.py
```

Options utiles:

```bash
python generate.py --prompt "Paris est" --max-new-tokens 60 --temperature 0.8 --top-k 40
```

## Hyperparametres par defaut

- `vocab_size = 30000`
- `d_model = 256`
- `n_head = 8`
- `n_layers = 6`
- `context_length = 128`
- `batch_size = 8`
- `lr = 3e-4`
- `max_steps = 30000`

## Limites actuelles

- seule une fraction du corpus est chargee en memoire
- pas de dataloader streaming
- pas de scheduler de learning rate
- pas encore de benchmark ou d'evaluation fixe par prompts
- qualite encore limitee pour un usage assistant

## Suite logique

Les prochains gains utiles sont:

- rendre l'entrainement plus robuste encore, avec scheduler et logs persistants
- comparer plusieurs tailles de vocabulaire et de `context_length`
- ajouter une evaluation reproductible sur une liste de prompts fixes
- passer ensuite a un fine-tuning instruction/chat
