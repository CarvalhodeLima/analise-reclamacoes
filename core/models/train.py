# core/models/make_model.py
import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------- paths ----------
BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # raiz do repo
DATA = os.path.join(BASE, "data", "train.ft.txt")                   # caminho do seu fastText
OUT_DIR = os.path.join(BASE, "notebooks")                           # onde salvar os PKLs
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- leitura fastText ----------
def read_fasttext(path, nrows=None):
    # formato: __label__1 texto...
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            label = parts[0].replace("__label__", "")
            text  = parts[1]
            rows.append((label, text))
    return pd.DataFrame(rows, columns=["label", "text"])

# ---------- pré-processamento igual ao do app ----------
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[^a-z0-9\sáéíóúãõâêôç]", " ", t)   # mantenha seus acentos/caráteres
    t = re.sub(r"\s+", " ", t).strip()
    return t

if __name__ == "__main__":
    # 1) ler dados (você pode amostrar com nrows=10000 para acelerar)
    df = read_fasttext(DATA, nrows=None)
    df["text"] = df["text"].apply(clean_text)

    # mapeie rótulos consistentemente com o app
    # supondo 1=Negativo, 2=Positivo (ajuste conforme seu dataset)
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 2) vetor + modelo (pipeline ou objetos separados)
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,2))
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, y_train)

    # 3) avaliação rápida
    y_pred = model.predict(Xte)
    print(classification_report(y_test, y_pred, digits=3))
    print(confusion_matrix(y_test, y_pred))

    # 4) salvar artefatos
    with open(os.path.join(OUT_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(OUT_DIR, "modelo_treinado.pkl"), "wb") as f:
        pickle.dump(model, f)

    print("Artefatos salvos em:", OUT_DIR)
