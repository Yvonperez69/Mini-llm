# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Mini-LLM causal écrit from scratch en PyTorch, entraîné sur un dump Wikipédia FR. Projet
pédagogique et mono-auteur : le but est de comprendre le pipeline complet sans librairie
d'entraînement haut niveau. La contribution originale est la couche **SCA** (Spectral
Cumulative Attention), comparée au Transformer pur à budget de paramètres égal.

Le code et les commentaires sont en français. Les commits aussi (souvent préfixés `J1`…`J7`
pour le jour du planning hebdomadaire).

## Interpréteur

`python3` sur le PATH est le Homebrew 3.14 et **n'a pas torch**. L'environnement du projet
est le conda base :

```bash
/opt/miniconda3/bin/python train.py     # torch 2.8.0, tokenizers 0.22.2
```

Toujours utiliser ce chemin (ou activer `conda activate base`). Un `python3 train.py` échoue
sur `ModuleNotFoundError: No module named 'torch'` et n'indique pas un vrai problème de code.

Dépendances : `torch matplotlib tokenizers mwparserfromhell numpy tensorboard`.

## Commandes

```bash
/opt/miniconda3/bin/python data/prepare_data.py    # data/input.txt → train.txt / val.txt (90/10)
/opt/miniconda3/bin/python data/split_articles.py  # stats tokens/article + filtered_train.txt
/opt/miniconda3/bin/python data/categories.py      # → logs/categories_freq.csv
/opt/miniconda3/bin/python tokenizer.py --vocab-size 16000 --input data/train.txt
/opt/miniconda3/bin/python train.py                # config en dur dans main(), pas de CLI
/opt/miniconda3/bin/python generate.py             # prompt interactif sur stdin
tensorboard --logdir runs/
```

Il n'y a **pas de suite de tests** ni de linter. `test.py` est un scratch gitignoré. La façon
de valider un changement de modèle est le smoke test : overfitter un seul batch et vérifier
que la loss tombe vers zéro en ~20 steps, en partant de `log(vocab_size)` au premier step.

Seul `tokenizer.py` a une interface argparse. `train.py` se configure en éditant le bloc
`# 1. config` de `main()` — et notamment `LOG_PATH` / `CHECKPOINT_PATH` / `SummaryWriter`,
qui sont en dur et **s'écrasent d'un run à l'autre si on ne les renomme pas**.

## État cassé actuel (vérifié)

Le refactor « SCA sans state » (commit `73576cb`) a changé la signature de
`Transformer.forward` sans mettre à jour les appelants. Quatre ruptures constatées :

| Symptôme | Cause |
| --- | --- |
| `train.py` : `AttributeError: 'tuple' object has no attribute 'shape'` | `compute_loss` fait `logits = model(inputs)` alors que `forward` renvoie toujours un tuple `(logits, states)` |
| `model/hybrid_block.py` : `'tuple' object has no attribute 'pow'` | `HybridBlock.forward` passe la sortie de `SCA` (un tuple) au `TransformerBlock`. Code mort — `transformer.py` n'utilise pas `HybridBlock` |
| `tokenizer.py` : `ModuleNotFoundError: No module named 'project_utils'` | le module importé n'existe pas dans le repo |
| `generate.py` : chemin `param/best_param_final.pt` | seul `param/best_param.pt` existe sur disque |

À cela s'ajoute un piège silencieux : `forward` renvoie `x, new_states if states is not None
else None`. Au premier appel de `generate.py`, `states=None` donc `new_states` revient à
`None` — la boucle de génération repart sans état à chaque token et ne voit qu'un token de
contexte. Symétriquement, `SCA.forward` fait `new_state = state` dans la branche training,
donc renvoie `None`. La récurrence O(1) annoncée dans le README ne s'active jamais telle
quelle.

## Architecture

Pipeline : dump Wikipédia → `data/input.txt` → `prepare_data.py` (split brut par offset de
caractères) → `train.txt` / `val.txt` → tokenizer BPE byte-level → `train.py` → checkpoint
`param/*.pt` → `generate.py`.

**Un article = une ligne** dans `train.txt`. C'est l'invariant sur lequel reposent
`split_articles.py` et `categories.py`. En revanche `train.py` ignore complètement cette
frontière : `get_data_iter` concatène les tokens de tous les articles dans un buffer plat et
découpe des fenêtres de `context_length + 1`. Un exemple d'entraînement chevauche donc
plusieurs articles. C'est correct pour du LM causal, mais incompatible avec la tâche de
classification par article en cours de construction — il faudra un second dataloader.

`text_fraction` (0.5) ne lit qu'une fraction des octets du fichier, pas un échantillon
aléatoire : c'est le début du fichier.

### Empilement des couches

`Transformer._build_layers` produit une `ModuleList` **plate**, pas des blocs hybrides. Pour
chaque itération de `n_layers`, il ajoute `sca_ratio` couches `SCA` puis un
`TransformerBlock`. Donc `n_layers=8, sca_ratio=1` → **16 modules**, ~59 M paramètres. Le
`sca_ratio` du README (1:1, 2:1) désigne ce ratio, et `n_layers` n'est pas le nombre réel de
couches. Le `forward` discrimine par `isinstance(layer, SCA)` pour savoir s'il doit passer et
récupérer un état.

### SCA (`model/sca.py`)

Attention linéaire à mémoire cumulative dans un domaine spectral, en 6 étapes commentées :
conv depthwise causale → projection et split en mémoire `(k, s)` et requêtes complexes
`(q_re, q_im)` → poids `alpha` → encodage en phase `(r, i)` → **accumulation** → lecture →
GatedRMSNorm + projection.

L'étape 4 est le cœur et a deux modes :

- `state is None` → `torch.cumsum` sur l'axe temporel. Mode entraînement, parallèle sur T.
- `state = (R, I, Z)` → addition incrémentale. Mode inférence, coût constant par token.

`R_hat`/`I_hat` sont normalisés par `Z`, ce qui rend les deux modes équivalents. La fonction
de décroissance temporelle `d` est **codée en dur à 0** (`d = 0`, ligne ~47) : `exp(-lam*d)`
vaut donc toujours 1 et `log_lambda` est un paramètre mort. L'activer est un chantier
explicite du README.

Contrainte : `n_head_memory == n_head_query` (assert).

### Pièges des hyperparamètres

- **`d_ff` est ignoré.** `FeedForward.__init__` accepte `d_ff` puis l'écrase par
  `round(8/3 * d_model / 64) * 64`. Passer `d_ff=4*d_model` depuis `train.py` ne fait rien :
  la valeur réelle est 1344 pour `d_model=512`. Le `d_ff` sauvé dans le checkpoint est donc
  mensonger.
- **Le cache RoPE est un buffer de taille fixe** construit dans `MultiHeadAttention.__init__`
  pour `context_length`. Augmenter le contexte à l'inférence au-delà de la valeur
  d'entraînement fait sortir des bornes de `self.cos`/`self.sin`.
- **Le checkpoint est incomplet.** `train.py` sauve `d_model`, `n_head`, `n_layers`,
  `sca_ratio`, `context_length`, `d_ff` — mais pas `n_kv_head`, `n_head_memory`,
  `n_head_query`, `spectral_sample` ni `head_dim`. `generate.py` retombe sur des `.get(...)`
  par défaut qui coïncident avec la config actuelle par chance. Changer un de ces
  hyperparamètres sans étendre le dict de sauvegarde produira un `load_state_dict` qui
  échoue.
- L'attention construit son masque causal à chaque forward (`torch.tril`) au lieu d'un
  buffer, et n'utilise pas `scaled_dot_product_attention`.

## Travail en cours

`semaine_01_classification.html` est le planning de la semaine — le lire pour comprendre la
direction. Objectif : passer du LM pur à la **classification de documents longs**, pour
donner à la SCA une tâche où son avantage sur le contexte long est mesurable. Décisions de la
semaine : `context_length` (viser le p75 des longueurs en tokens), ~20 catégories macro
mono-label, tokenizer à 16 000, bf16 et plafond mémoire 12 Go, puis pré-entraînement LM
avant tout fine-tuning.

`data/categories.py` est à cheval sur deux étapes et contient actuellement **deux blocs qui
se recouvrent** : l'extraction fonctionnelle (mercredi) et un début de taxonomie figée
`MACRO_CATEGORIES` / `LABEL2ID` collé après le `main()` (jeudi), qui référence un
`category_mapping.csv` encore absent. À nettoyer avant d'aller plus loin.

Sur l'extraction, deux points non évidents découverts dans le dump :

- Les catégories sont **accolées, séparées par une simple espace, sans délimiteur** :
  `Catégorie:Matériel Apple Catégorie:Ordinateur 8 bits`. Il n'y a rien sur quoi `split()` ;
  d'où le lookahead non-greedy `(.+?)(?={MARKER}|$)` de `CAT_RE`.
- Sur le dernier marqueur d'une ligne ce lookahead retombe sur `$` et avale la prose. Le dump
  contient des pages de discussion Wikipédia où cela capture jusqu'à 541 ko dans un seul
  « nom de catégorie ». D'où `MAX_LABEL_LEN = 80`, calibré sur la distribution mesurée
  (médiane 26, p99 69).

Le marqueur doit impérativement être retiré du texte avant entraînement : le laisser
donnerait la réponse au modèle et invaliderait tous les scores.

## Fichiers non versionnés

`.gitignore` exclut les corpus (`data/*.txt` via `**/*.txt`), `param/`, `runs/`,
`output.txt`, `test.py`. Ne pas s'attendre à trouver `data/train.txt` sur un clone frais.
Attention : `**/*.txt` couvre tout le repo, mais **pas** `logs/*.csv` — le
`categories_freq.csv` de 12 Mo sera versionné s'il n'est pas exclu explicitement.

## Commits

Ne pas ajouter de trailer `Co-Authored-By: Claude`. Le projet est mono-auteur et le crédit de
co-auteur fausserait sa paternité.
