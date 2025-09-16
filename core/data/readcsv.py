import pandas as pd

def load_fasttext(path, nrows=None):
    """
    Lê arquivos em formato FastText (.ft.txt)
    Retorna DataFrame com colunas 'label' e 'text'.
    """
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                label = parts[0].replace("__label__", "")
                text = parts[1]
                data.append((label, text))
    return pd.DataFrame(data, columns=["label", "text"])
