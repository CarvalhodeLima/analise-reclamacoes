import sqlite3
import pandas as pd
from core.data.readcsv import load_fasttext

print("Arquivo database.py foi carregado")
print("__name__ =", __name__)

DB_NAME = "reviews.db"

def create_connection():
    """Cria conexão com o banco SQLite."""
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_table():
    """Cria tabela SOR (dados crus)."""
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sor_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label INTEGER,
            text TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_data(df: pd.DataFrame):
    """Insere DataFrame no SOR."""
    conn = create_connection()
    df.to_sql("sor_reviews", conn, if_exists="append", index=False)
    conn.close()

def load_sor(limit=5):
    """Carrega dados da tabela SOR (para testar)."""
    conn = create_connection()
    df = pd.read_sql(f"SELECT * FROM sor_reviews LIMIT {limit}", conn)
    conn.close()
    return df

if __name__ == "__main__":
    print("Iniciando teste do database.py...")

    create_table()
    print("Tabela criada com sucesso.")

    df = load_fasttext("data/train.ft.txt", nrows=5)  
    print("Dados carregados do CSV:", df.shape)

    insert_data(df)
    print("Dados inseridos no banco.")

    print("Primeiras linhas do SOR:")
    print(load_sor())
