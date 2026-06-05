# A small-scale Hybrid SCA-Transformer language model

Mini-LLM causal code from scratch en PyTorch, entraîné sur un corpus texte francophone et capable de générer des complétions simples.

Le projet a un objectif pédagogique : comprendre la chaîne complète d'un modèle de langage sans dépendre d'une grosse librairie de training. On y trouve :

- un Transformer causal implémenté à la main
- implémentation d'une couche SCA
- un tokenizer BPE entraîné sur le corpus
- une boucle d'entraînement simple
- un script de génération avec inférence récurrente 

## Aperçu

Le modèle apprend à prédire le token suivant à partir d'un contexte de longueur fixe. L'architecture repose sur :

- embeddings de tokens
- encodage positionnel rotatif RoPE
- Grouped Query Attention GQA
- feed-forward network SwiGLU
- projection finale vers le vocabulaire
- scheduler cosine pour le learning rate

Le dépôt est volontairement compact pour rester lisible et modifiable.

## Structure

```text
Mini-llm/
├── model/
│   ├── attention.py
│   ├── block.py
│   ├── feedforward.py
│   ├── norms.py
│   ├── sca.py
│   └── transformer.py
├── data/
│   └── prepare_data.py
├── logs/
│   ├── graph.py
│   ├── metrics_A.csv
│   ├── metrics_B.csv
│   ├── metrics_C.csv
│   └── metrics_final.csv
├── tokenizer.py
├── train.py
├── generate.py
├── README.md
└── .gitignore
```

## Installation

Ce projet suppose un environnement Python avec les dépendances suivantes :

```bash
pip install torch matplotlib tokenizers mwparserfromhell
```

## Utilisation

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

## Résultats 
On a testé plusieurs ratio entre couche SCA et Transformer afin de comparer les performances à nombre de parametres égales.

| Run | Config | Params | Val. loss |
| --- | --- | --- | --- |
| A | Hybrid 1:1 | 55M | 3.109 |
| B | Hybrid 2:1 | 52M | 3.124 |
| C | Pure Transformer | 56M | 3.236 |

## Ce que le projet fait bien

- montrer clairement comment fonctionne un petit LLM causal
- permettre des expérimentations rapides sur l'architecture et la génération
- servir de base de travail pour progresser vers un modèle plus propre

## Limites actuelles



## Pistes d'amélioration



## Objectif du projet

Ce dépôt vise surtout à apprendre :

