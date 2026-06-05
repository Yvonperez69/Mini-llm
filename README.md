# A Small-Scale Hybrid SCA-Transformer Language Model

A causal Mini-LLM coded from scratch in PyTorch, trained on a French 
text corpus and capable of generating simple completions.

The project has a pedagogical objective: understand the full pipeline 
of a language model without relying on a high-level training library. 
It includes:

- a causal Transformer implemented from scratch
- a full SCA layer implementation
- a BPE tokenizer trained on the corpus
- a simple training loop
- a generation script with recurrent inference

## Overview

The model learns to predict the next token from a fixed-length context. 
The architecture relies on:

- token embeddings
- rotary position embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward network
- final projection to the vocabulary
- cosine learning rate scheduler with linear warmup

The repository is intentionally compact to remain readable and easy to modify.

## Structure
...

## Installation

This project requires a Python environment with the following dependencies:

```bash
pip install torch matplotlib tokenizers mwparserfromhell
```

## Usage

### Train the tokenizer

```bash
python tokenizer.py
```

### Train the model

```bash
python train.py
```

### Generate text

```bash
python generate.py
```

Then enter a prompt, for example:

```text
La meilleure ville de France est
```

## Results

We compared several SCA/Transformer ratios at equal parameter budget.

| Run | Config | Params | Val. loss |
| --- | --- | --- | --- |
| A | Hybrid 1:1 | 55M | 3.109 |
| B | Hybrid 2:1 | 52M | 3.124 |
| C | Pure Transformer | 56M | 3.236 |

Both hybrid models outperform the pure Transformer baseline at equal 
parameter budget. The hybrid 1:1 achieves the best validation loss.

## What this project does well

- clearly shows how a small causal LLM works end-to-end
- allows quick experimentation on architecture and generation
- serves as a working base to progress toward a cleaner model

## Current limitations

The model generates grammatically plausible French text but lacks 
semantic coherence, which is expected at this scale with limited 
parameters and training data.

## Future work

- Instruction fine-tuning — fine-tune the final checkpoint on 
  French instruction/response pairs
- Longer context — increase context_length from 256 to 512 or 1024
- Temporal decay in SCA — activate and measure the impact of d(t)

## Project goal

This repository is primarily a learning project:

- Understand the structure of language models, how to train them 
  and explore their limits
- Discover the world of ML/DL research
- Develop skills and knowledge in these areas