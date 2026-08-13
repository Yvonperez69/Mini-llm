from pathlib import Path
import numpy as np 

from tokenizers import Tokenizer 


train_path = Path(__file__).resolve().parent / "train.txt" 
val_path = Path(__file__).resolve().parent  / "val.txt" 
tokenizer_path = Path(__file__).resolve().parent.parent / "tokenizer.json" 
filtered_train_path = Path(__file__).resolve().parent / "filtered_train.txt" 
filtered_val_path = Path(__file__).resolve().parent / "filtered_val.txt" 

def split_articles(data_path) : 
    with open(data_path, "r", encoding='utf-8', errors='ignore') as f : 
        for line in f :
            yield line.strip() 

tokenizer = Tokenizer.from_file(str(tokenizer_path)) 
vocab_size = tokenizer.get_vocab_size() 

def token_lengths(data_path, tokenizer,limit = None, batch_size=1000) :
    batch = [] 
    lengths = []
    for i, article in enumerate(split_articles(data_path)) :
        if limit is not None and i >= limit :
            break
        batch.append(article)

        if len(batch) == batch_size :
            encodings = tokenizer.encode_batch(batch) 
            for enc in encodings :
                lengths.append(len(enc.ids))
            batch = [] 
    if batch :
        encodings = tokenizer.encode_batch(batch)
        for enc in encodings :
            lengths.append(len(enc.ids)) 
    return lengths  

lengths = token_lengths(train_path, tokenizer=tokenizer, limit=1)
print("nombre d'article ",len(lengths)) 
arr = np.array(lengths) 
print("percentile : ", np.percentile(arr, [50, 75, 90, 99])) 
print("moyenne : ", arr.mean()) 
print("min :", arr.min()) 
print("max :", arr.max()) 
for ctx_l in [256, 512, 1024] :
    print(f" proportion d'article restant avec context_length : {ctx_l}, \n {(arr <=ctx_l).mean():.1%} ") 

def get_article(data_path,tokenizer, min_tokens=1024, limit= None, batch_size =1000) :
    batch = [] 
    for i, article in enumerate(split_articles(data_path)) :
        if limit is not None and i >= limit :
            break
        batch.append(article)
        if len(batch) == batch_size : 
            encodings = tokenizer.encode_batch(batch) 
            for j, enc in enumerate(encodings) :
                if len(enc.ids) >= min_tokens :
                    yield batch[j] 
            batch = [] 
    if batch :
        encodings = tokenizer.encode_batch(batch) 
        for i, enc in enumerate(encodings) :
            if len(enc.ids) >= min_tokens :
                yield batch[i] 

def filtre_data(old_data_path, new_data_path, tokenizer) :
    with open(new_data_path, "w", encoding='utf-8') as f : 
        for line in get_article(old_data_path, tokenizer) :
            f.write(line + "\n")  

filtre_data(train_path,filtered_train_path,tokenizer) 
filtre_data(val_path,filtered_val_path,tokenizer) 

