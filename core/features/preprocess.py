# core/features/preprocess.py

import re
import pandas as pd

def clean_text(text):
    """
    Limpa e normaliza um texto:
    - minúsculas
    - remove caracteres não alfanuméricos (mantém espaços)
    - comprime espaços múltiplos
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def transform_dataframe(df):
    """
    Recebe DataFrame com colunas ['label','text'] e adiciona 'clean_text'.
    """
    df_out = df.copy()
    df_out["clean_text"] = df_out["text"].apply(clean_text)
    return df_out