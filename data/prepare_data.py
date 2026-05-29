from pathlib import Path

INPUT  = Path("data/input.txt")
TRAIN  = Path("data/train.txt")
VAL    = Path("data/val.txt")
SPLIT  = 0.9

with open(INPUT, "r") as f:
    text = f.read()

split = int(0.9 * len(text))
with open(TRAIN, "w") as f:
    f.write(text[:split])
with open(VAL, "w") as f:
    f.write(text[split:])

print(f"total : {len(text)/1e6:.1f}MB")
print(f"train : {split/1e6:.1f}MB → {TRAIN}")
print(f"val   : {(len(text)-split)/1e6:.1f}MB → {VAL}")