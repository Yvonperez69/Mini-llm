"""Chemins partagés du projet.

Reconstruit d'après les usages dans tokenizer.py — le fichier d'origine n'a
jamais été committé.

Intérêt : les chemins par défaut et les chemins relatifs passés en CLI sont
résolus par rapport à la racine du repo, pas au répertoire courant. Un
`python tokenizer.py --input data/train.txt` marche donc depuis n'importe où.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_CORPUS_PATH = ROOT_DIR / "data" / "input.txt"
DEFAULT_TOKENIZER_PATH = ROOT_DIR / "tokenizer.json"


def resolve_path(path) -> Path:
    """Résout un chemin relatif depuis la racine du repo (absolu inchangé)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def ensure_parent_dir(path) -> Path:
    """Crée le dossier parent de `path` s'il manque, et renvoie le chemin."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
