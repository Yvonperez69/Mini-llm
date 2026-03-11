from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# créer tokenizer vide
tokenizer = Tokenizer(BPE())

# byte-level pretokenizer
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
)

# entrainement sur ton dataset
files = ["/Users/yvonperez/Dropbox/Mac/Documents/Info/Attention/data/input.txt"]   # ton gros fichier texte
tokenizer.train(files, trainer)

# decoder
tokenizer.decoder = ByteLevelDecoder()

# sauvegarde
tokenizer.save("tokenizer.json")