from os import wait
from pathlib import Path 
from tokenizers import Tokenizer 
tokenizer_path = Path(__file__).resolve().parent / "tokenizer.json" 

tokenizer = Tokenizer.from_file(str(tokenizer_path)) 
print(tokenizer.get_vocab_size()) 

