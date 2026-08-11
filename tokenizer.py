import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from project_utils import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_TOKENIZER_PATH,
    ROOT_DIR,
    ensure_parent_dir,
    resolve_path,
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraine un tokenizer BPE byte-level sur le corpus."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_CORPUS_PATH.relative_to(ROOT_DIR)),
        help="Chemin vers le corpus texte",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TOKENIZER_PATH.relative_to(ROOT_DIR)),
        help="Chemin de sortie du tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Taille du vocabulaire cible",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Frequence minimale pour garder un token",
    )
    return parser.parse_args()


def train_tokenizer(
    input_path: Path,
    output_path: Path,
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train([str(input_path)], trainer)
    tokenizer.decoder = ByteLevelDecoder()

    ensure_parent_dir(output_path)
    tokenizer.save(str(output_path))
    return tokenizer


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Corpus introuvable: {input_path}")

    tokenizer = train_tokenizer(
        input_path=input_path,
        output_path=output_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(
        f"Tokenizer sauve dans {output_path} | vocab_size={tokenizer.get_vocab_size()}"
    )


if __name__ == "__main__":
    main()
