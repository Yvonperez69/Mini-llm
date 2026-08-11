from pathlib import Path

train_path = Path(__file__).resolve().parent / "train.txt" 
val_path = Path(__file__).resolve().parent  / "val.txt" 


def split_articles(data_path) : 
    with open(data_path, "r") as f : 
        articles = f.readlines()
    return articles 

print(len(split_articles(train_path))) 


