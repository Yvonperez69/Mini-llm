"""Extraction des marqueurs `Catégorie:` du corpus.

Un article = une ligne. Les catégories sont accolées en fin de ligne, séparées
par une simple espace et sans délimiteur :

    ... Liens externes Catégorie:Matériel Apple Catégorie:Ordinateur 8 bits

Il n'y a donc aucun caractère sur lequel découper : le nom d'une catégorie court
du marqueur jusqu'au marqueur suivant (ou la fin de ligne). D'où le lookahead
non-greedy de CAT_RE.

Sortie : logs/categories_freq.csv (categorie, nb_articles, type)
"""

import csv
import re
from collections import Counter
from pathlib import Path
import json 

DATA_DIR = Path(__file__).resolve().parent
LOGS_DIR = DATA_DIR.parent / "logs"
CORPUS_PATHS = [DATA_DIR / "filtered_train.txt", DATA_DIR / "filtered_val.txt"]
FREQ_CSV = LOGS_DIR / "categories_freq.csv"
data_w_lab = DATA_DIR / "train_w_lab.jsonl" 
# Le marqueur lui-même. Tolère les variantes vues dans le dump :
#   "Catégorie:" (1 017 091 occurrences), "Catégorie :" (88), "catégorie:" (983),
#   "Categorie:" (5), et le "*" de puce wiki qui précède parfois le bloc.
MARKER = r"[*\s]*[Cc]at[ée]gorie\s*:\s*"

# Le nom : tout jusqu'au marqueur suivant ou la fin de ligne.
CAT_RE = re.compile(rf"{MARKER}(.+?)(?={MARKER}|$)")

# Garde-fou. Sur le dernier marqueur d'une ligne le lookahead retombe sur `$`
# et avale la prose restante — cas des pages de discussion présentes dans le
# dump, où l'on capture jusqu'à 541 ko de texte. Mesuré sur train.txt :
# médiane 26, p90 45, p99 69 ; 99,4 % des segments tiennent sous 80.
MAX_LABEL_LEN = 80

# Catégories de maintenance / méta : le label ne dit rien du sujet de l'article.
ADMIN_RE = re.compile(
    r"""^(?:
          Portail | Wikip[ée]dia | Projet | Mod[èe]le | Utilisateur | Aide
        | Cat[ée]gorie | Discussion | Page | Pages | Article | Articles
        | [ÉE]bauche | Homonymie | Multi\ bandeau | Bon\ article
        )\b
    | \b(?:
          [àa]\ sourcer | [àa]\ v[ée]rifier | [àa]\ recycler | [àa]\ wikifier
        | [àa]\ illustrer | [àa]\ fusionner | [àa]\ relire | [àa]\ traduire
        | sans\ source | non\ r[ée]f[ée]renc | non\ relue | orphelin
        | notice\ d'autorit[ée] | appel\ [àa]\ traduction
        | non\ renseign | inconnue? \s*$
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Axes biographiques génériques : thématiquement vides ("Décès à 77 ans"), mais
# ils saturent le top 100. Isolés pour ne pas les confondre avec du vrai label.
BIO_RE = re.compile(
    r"^(?:Naissance|D[ée]c[èe]s|Mort)\s+(?:[àa]|en|dans|le)\b"
    r"|^(?:Date|Lieu)\s+de\s+(?:naissance|d[ée]c[èe]s|mort)\b",
    re.IGNORECASE,
)


def clean_label(raw: str) -> str:
    """Normalise un nom capturé, ou renvoie "" s'il n'est pas exploitable."""
    label = raw.split("|", 1)[0]  # clé de tri wiki : "Catégorie:Altiste|Nom,"
    label = label.replace("[[", " ").replace("]]", " ")
    label = re.sub(r"\s+", " ", label).strip(" *.,;:!?…\"'")
    # Beaucoup de labels finissent par une parenthèse utile — "Anoure (nom
    # scientifique)" — donc on ne la rogne pas ; on ne répare que les
    # parenthèses orphelines laissées par une capture tronquée.
    if label.count("(") > label.count(")"):
        label = label[: label.rindex("(")].strip(" *.,;:")
    label = label.rstrip(")") if label.count(")") > label.count("(") else label
    if not label or len(label) > MAX_LABEL_LEN:
        return ""
    return label


def label_type(label: str) -> str:
    if ADMIN_RE.search(label):
        return "admin"
    if BIO_RE.search(label):
        return "bio"
    return "thematique"


def extract_categories(text: str) -> tuple[list[str], str]:
    """Renvoie (labels retenus, texte débarrassé des marqueurs).

    Un segment rejeté (trop long = prose avalée) voit son marqueur supprimé
    mais son texte conservé : on ne veut pas amputer l'article.
    """
    labels, kept, cursor = [], [], 0
    for m in CAT_RE.finditer(text):
        label = clean_label(m.group(1))
        kept.append(text[cursor : m.start()])
        if label:
            labels.append(label)
        else:
            # On garde la prose, pas le marqueur. L'espace initial est réinjecté
            # car `[*\s]*` l'a consommé et il sépare deux mots de la phrase.
            kept.append(" " + m.group(1))
        cursor = m.end()
    kept.append(text[cursor:])
    cleaned = re.sub(r"\s+", " ", "".join(kept)).strip()
    return labels, cleaned

def clean_text(text):
    kept, cursor = [], 0
    for m in CAT_RE.finditer(text):
        # 1. On garde la prose AVANT le match
        kept.append(text[cursor : m.start()])
        # 2. On ignore le label (valide ou invalide)
        cursor = m.end()
    # 3. On ajoute le texte restant après le dernier match
    kept.append(text[cursor:])
    # 4. Nettoyage des espaces multiples
    cleaned = re.sub(r"\s+", " ", "".join(kept)).strip()
    return cleaned

def strip_categories(text: str) -> str:
    return extract_categories(text)[1]


def count_corpus(paths) -> tuple[Counter, int, int]:
    """Compte en nombre d'ARTICLES : une catégorie répétée dans une même ligne
    ne compte qu'une fois."""
    freq = Counter()
    n_articles = n_labelled = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_articles += 1
                labels, _ = extract_categories(line)
                if labels:
                    n_labelled += 1
                    freq.update(set(labels))
    return freq, n_articles, n_labelled


def main() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    freq, n_articles, n_labelled = count_corpus(CORPUS_PATHS)

    rows = [(lab, n, label_type(lab)) for lab, n in freq.most_common()]
    with open(FREQ_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["categorie", "nb_articles", "type"])
        w.writerows(rows)

    per_type = Counter(t for _, _, t in rows)
    print(f"articles                : {n_articles}")
    print(f"articles avec catégorie : {n_labelled} ({n_labelled / n_articles:.1%})")
    print(f"catégories distinctes   : {len(rows)}")
    for t, n in per_type.most_common():
        print(f"  {t:11s} : {n:6d} labels distincts")
    print(f"→ {FREQ_CSV}")

    # La queue est énorme : sans seuil il n'y a pas de classes apprenables.
    print("\nlabels thématiques au-dessus d'un seuil d'articles :")
    thematic = [(lab, n) for lab, n, t in rows if t == "thematique"]
    for seuil in (10, 50, 100, 500, 1000):
        gardes = [n for _, n in thematic if n >= seuil]
        print(f"  ≥ {seuil:5d} articles : {len(gardes):6d} labels, {sum(gardes):8d} paires (article, label)")

    print("\n=== TOP 100 ===")
    for lab, n, t in rows[:100]:
        print(f"{n:6d}  [{t:10s}]  {lab}")
# data/categories.py
"""Taxonomie figée pour la classification — décidée le [date].
Voir category_mapping.csv pour le mapping complet catégorie fine → macro."""

# 1. La liste des classes retenues, dans un ordre fixe
MACRO_CATEGORIES = [
    "cinema_tv",
    "sport",
    "litterature",
    "geographie",
    "arts_visuels",
    "musique",
    "politique",
    "jeu_video",
    "militaire",
    "transport",      # ou pas, selon ta décision
    "histoire",
    "sciences",
]

# 2. Le mapping label → index, pour la loss et la tête de classification
LABEL2ID = {c: i for i, c in enumerate(MACRO_CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

# 3. Le chargement du mapping fin → macro depuis le CSV
import csv
from pathlib import Path

def load_mapping(path=Path(__file__).parent / "category_mapping.csv"):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["macro"] != "IGNORE" and row["macro"] in LABEL2ID:
                mapping[row["categorie"]] = row["macro"]
    return mapping
# on fait un json avec le texte et le label 
def load_data(data_path) :
    with open(data_path, "r") as f : 
        for line in f : 
            yield line 

def save_data_with_label(data_path) :
    
    mapping = load_mapping() 
    with open(data_w_lab, "w") as f :
        for article in load_data(data_path) :
            cat = extract_categories(article) 
            if cat[0] and cat[0][0] in mapping : 
                print(len(cat[0])) 
                label = mapping[cat[0][0]] 
                f.write(json.dumps({"article": clean_text(article), "label": label}, ensure_ascii=False) + "\n") 
            

if __name__ == "__main__":
    save_data_with_label(CORPUS_PATHS[0]) 
