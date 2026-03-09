
class Tokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
    
    def build_vocab(self, text):
        words = text.split()
        vocab = sorted(set(words))
        
        self.word2id = {w:i for i, w in enumerate(vocab)}
        self.id2word = {i:w for w, i in self.word2id.items()}
    
    def encode(self, text):
        return [self.word2id[w] for w in text.split() if w in self.word2id]
    
    def decode(self, idx):
        return " ".join(self.id2word[i] for i in idx)