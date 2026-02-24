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


