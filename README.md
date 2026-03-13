# Attention

Mini-LLM causal code from scratch en PyTorch, entraîne sur un corpus texte francophone et capable de générer des complétions simples.

Le projet a un objectif pédagogique : comprendre la chaîne complète d'un modèle de langage sans dépendre d'une grosse librairie de training. On y trouve :

- un Transformer causal implémenté à la main
- un tokenizer BPE entraîné sur le corpus
- un script d'extraction de texte depuis un dump Wikipédia
- une boucle d'entraînement simple
- un script de génération avec sampling

## Aperçu

Le modèle apprend à prédire le token suivant à partir d'un contexte de longueur fixe. L'architecture repose sur :

- embeddings de tokens
- encodage positionnel sinusoïdal
- multi-head self-attention causale
- feed-forward network
- projection finale vers le vocabulaire

Le dépôt est volontairement compact pour rester lisible et modifiable.

## Structure

```text
.
├── model/
│   ├── attention.py
│   ├── block.py
│   ├── feedforward.py
│   └── transformer.py
├── extract_wiki_txt.py
├── tokenizer.py
├── train.py
├── generate.py
└── tokenizer.json
```

## Pipeline

### 1. Préparer le corpus

`extract_wiki_txt.py` extrait du texte brut depuis un dump Wikipédia compressé `.bz2` en supprimant une partie du markup.

Le script écrit actuellement dans `wiki_fr.txt`, alors que `train.py` et `tokenizer.py` attendent un corpus dans `data/input.txt`.

Deux options simples :

1. Modifier `OUTPUT_FILE` dans `extract_wiki_txt.py` pour écrire directement dans `data/input.txt`
2. Déplacer ou renommer le fichier extrait vers `data/input.txt`

### 2. Entraîner le tokenizer

`tokenizer.py` entraîne un tokenizer BPE byte-level avec `tokenizers` puis sauvegarde le résultat dans `tokenizer.json`.

Configuration actuelle :

- `vocab_size = 30000`
- `min_frequency = 2`
- tokens spéciaux : `<pad>`, `<unk>`, `<bos>`, `<eos>`

Pour une machine personnelle, un vocabulaire de `8000` à `16000` est souvent plus raisonnable.

### 3. Entraîner le modèle

`train.py` :

- charge une fraction du corpus pour éviter de saturer la mémoire
- tokenize le texte
- crée un split train/validation
- entraîne le Transformer en next-token prediction
- sauvegarde le meilleur checkpoint dans `param/best_param.pt`

Hyperparamètres actuels :

- `TEXT_FRACTION = 0.01`
- `d_model = 256`
- `n_head = 8`
- `n_layers = 6`
- `context_length = 128`
- `batch_size = 8`
- `lr = 3e-4`
- `max_steps = 30000`

### 4. Générer du texte

`generate.py` recharge le checkpoint et génère une complétion à partir d'un prompt fourni via l'entrée standard.

La génération utilise :

- `temperature`
- `top_k`
- `repetition_penalty`
- arrêt sur `<eos>` si disponible

## Installation

Ce projet suppose un environnement Python avec les dépendances suivantes :

```bash
pip install torch matplotlib tokenizers mwparserfromhell
```

## Utilisation

### Extraire un corpus

Place un dump Wikipédia dans le dépôt, par exemple `frwiki-latest-pages-articles.xml.bz2`, puis lance :

```bash
python extract_wiki_txt.py
```

### Entraîner le tokenizer

```bash
python tokenizer.py
```

### Entraîner le modèle

```bash
python train.py
```

### Générer du texte

```bash
python generate.py
```

Puis saisis un prompt, par exemple :

```text
La meilleure ville de France est
```

## Ce que le projet fait bien

- montrer clairement comment fonctionne un petit LLM causal
- permettre des expérimentations rapides sur l'architecture et la génération
- servir de base de travail pour progresser vers un modèle plus propre

## Limites actuelles

- entraînement sur une petite fraction du corpus
- pas de dataloader streaming
- pas de scheduler de learning rate
- pas de gradient clipping
- pas de seed globale pour la reproductibilité
- pas d'instruction tuning ni de dataset conversationnel

En pratique, le modèle peut produire du texte plausible, mais ce n'est pas encore un assistant conversationnel fiable.

## Pistes d'amélioration

- réduire et nettoyer davantage le corpus
- utiliser un vocabulaire plus petit et mieux adapté
- ajouter un vrai fallback `cpu` dans `train.py`
- rendre l'entraînement plus robuste
- ajouter des métriques et prompts d'évaluation fixes
- fine-tuner ensuite sur des données instruction/chat

## Objectif du projet

Ce dépôt vise surtout à apprendre :

- comment fonctionne un Transformer causal
- comment préparer un corpus
- comment entraîner un tokenizer
- comment entraîner puis utiliser un mini modèle de langage

Si l'objectif est de construire un vrai assistant, ce repo est une bonne base de compréhension, pas encore un produit final.
