import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer

from model.transformer import Transformer
from project_utils import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_TOKENIZER_PATH,
    ROOT_DIR,
    ensure_parent_dir,
    resolve_device,
    resolve_path,
    set_seed,
)

import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraine le mini Transformer causal.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_CORPUS_PATH.relative_to(ROOT_DIR)),
        help="Chemin vers le corpus texte",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_TOKENIZER_PATH.relative_to(ROOT_DIR)),
        help="Chemin vers tokenizer.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH.relative_to(ROOT_DIR)),
        help="Chemin de sauvegarde du checkpoint",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou mps")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.01,
        help="Fraction du corpus chargee en memoire",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed globale")
    parser.add_argument("--d-model", type=int, default=256, help="Dimension du modele")
    parser.add_argument("--n-head", type=int, default=8, help="Nombre de tetes")
    parser.add_argument("--n-layers", type=int, default=6, help="Nombre de blocs")
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        help="Longueur maximale de contexte",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay AdamW"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30000,
        help="Nombre de pas d'optimisation a executer",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Frequence d'evaluation sur la validation",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=200,
        help="Nombre de batches utilises pour estimer la val_loss",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Frequence d'affichage de la train_loss",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Norme max du gradient, <=0 pour desactiver",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Recharge et continue le checkpoint existant",
    )
    return parser.parse_args()


def read_text_sample(data_path: Path, fraction: float) -> tuple[str, int, int]:
    if not 0 < fraction <= 1:
        raise ValueError("--fraction doit etre strictement compris entre 0 et 1.")

    data_size = data_path.stat().st_size
    sample_size = max(1, int(data_size * fraction))

    with data_path.open("rb") as source:
        text_bytes = source.read(sample_size)

    text = text_bytes.decode("utf-8", errors="ignore")
    return text, sample_size, data_size


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> torch.Tensor:
    if len(tokens) <= context_length:
        raise ValueError(
            "Le corpus tokenise est trop petit pour le context_length choisi."
        )

    start_indices = torch.randint(0, len(tokens) - context_length, (batch_size,))
    batch = torch.stack(
        [tokens[start : start + context_length + 1] for start in start_indices]
    )
    return batch.to(device)


def compute_loss(
    model: Transformer,
    idx: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    inputs = idx[:, :-1]
    targets = idx[:, 1:]
    logits = model(inputs)
    batch_size, time_steps, vocab_size = logits.shape
    logits = logits.reshape(batch_size * time_steps, vocab_size)
    targets = targets.reshape(batch_size * time_steps)
    return criterion(logits, targets)


def training_step(
    model: Transformer,
    optimizer: optim.Optimizer,
    idx: torch.Tensor,
    criterion: nn.Module,
    grad_clip: float,
) -> float:
    loss = compute_loss(model, idx, criterion)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    optimizer.step()
    return loss.item()


def estimate_val_loss(
    model: Transformer,
    val_tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    criterion: nn.Module,
    device: torch.device,
    eval_iters: int,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(eval_iters):
            idx = get_batch(val_tokens, batch_size, context_length, device)
            total_loss += compute_loss(model, idx, criterion).item()

    model.train()
    return total_loss / eval_iters


def build_model(
    vocab_size: int,
    d_model: int,
    n_head: int,
    n_layers: int,
    d_ff: int,
    device: torch.device,
) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_head=n_head,
        n_layers=n_layers,
    )
    return model.to(device)


def load_training_state(
    checkpoint_path: Path,
    device: torch.device,
) -> dict | None:
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = resolve_path(args.data)
    tokenizer_path = resolve_path(args.tokenizer)
    checkpoint_path = resolve_path(args.checkpoint)
    device = resolve_device(args.device)

    if not data_path.exists():
        raise FileNotFoundError(f"Corpus introuvable: {data_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer introuvable: {tokenizer_path}")

    text, sample_size, data_size = read_text_sample(data_path, args.fraction)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = tokenizer.encode(text)
    tokens = torch.tensor(encoding.ids, dtype=torch.long)
    vocab_size = tokenizer.get_vocab_size()

    checkpoint = None
    d_model = args.d_model
    n_head = args.n_head
    n_layers = args.n_layers
    d_ff = 4 * d_model
    context_length = args.context_length
    best_val_loss = math.inf
    start_step = 0

    if args.resume:
        checkpoint = load_training_state(checkpoint_path, device)
        if checkpoint is None:
            raise FileNotFoundError(
                f"--resume est active mais aucun checkpoint n'existe: {checkpoint_path}"
            )

        checkpoint_vocab_size = checkpoint["vocab_size"]
        if checkpoint_vocab_size != vocab_size:
            raise ValueError(
                "Le tokenizer courant est incompatible avec le checkpoint: vocab_size differente."
            )

        d_model = checkpoint["d_model"]
        n_head = checkpoint["n_head"]
        n_layers = checkpoint["n_layers"]
        d_ff = checkpoint.get("d_ff", 4 * d_model)
        context_length = checkpoint["context_length"]
        best_val_loss = checkpoint.get("best_val_loss", math.inf)
        start_step = checkpoint.get("step", -1) + 1

    if len(tokens) < context_length + 2:
        raise ValueError(
            "Le nombre de tokens est insuffisant. Augmente --fraction ou reduis --context-length."
        )

    train_split = int(0.9 * len(tokens))
    train_tokens = tokens[:train_split]
    val_tokens = tokens[train_split:]

    if len(val_tokens) < context_length + 2:
        raise ValueError(
            "La validation est trop petite. Augmente --fraction ou reduis --context-length."
        )

    model = build_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        d_ff=d_ff,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    end_step = start_step + args.max_steps

    print(
        f"device={device} | sample={sample_size / 1024 / 1024:.1f}MB/{data_size / 1024 / 1024:.1f}MB "
        f"| vocab_size={vocab_size} | train_tokens={len(train_tokens)} | val_tokens={len(val_tokens)}"
    )
    print(
        f"config | d_model={d_model} n_head={n_head} n_layers={n_layers} "
        f"context_length={context_length} batch_size={args.batch_size}"
    )

    model.train()
    for step in range(start_step, end_step):
        idx = get_batch(train_tokens, args.batch_size, context_length, device)
        loss = training_step(
            model=model,
            optimizer=optimizer,
            idx=idx,
            criterion=criterion,
            grad_clip=args.grad_clip,
        )

        if step == start_step or (step + 1) % args.log_interval == 0:
            print(f"step {step + 1}/{end_step} | train_loss={loss:.4f}")

        should_eval = (step == start_step) or ((step + 1) % args.eval_interval == 0)
        if should_eval:
            val_loss = estimate_val_loss(
                model=model,
                val_tokens=val_tokens,
                batch_size=args.batch_size,
                context_length=context_length,
                criterion=criterion,
                device=device,
                eval_iters=args.eval_iters,
            )
            print(f"step {step + 1}/{end_step} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ensure_parent_dir(checkpoint_path)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "vocab_size": vocab_size,
                        "d_model": d_model,
                        "d_ff": d_ff,
                        "n_head": n_head,
                        "n_layers": n_layers,
                        "context_length": context_length,
                        "best_val_loss": best_val_loss,
                        "step": step,
                        "tokenizer_path": str(tokenizer_path),
                    },
                    checkpoint_path,
                )
                print(
                    f"checkpoint saved | path={checkpoint_path} | best_val_loss={best_val_loss:.4f}"
                )


if __name__ == "__main__":
    main()
