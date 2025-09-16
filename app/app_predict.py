import pickle
import os

# Caminho base para os arquivos salvos no notebooks/
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")

MODEL_PATH = os.path.join(BASE_DIR, "modelo_treinado.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# === carregar modelo e vectorizer ===
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# === loop de interação ===
print("=== Analisador de Sentimentos ===")
print("Digite uma frase para classificar (ou 'sair' para encerrar):")

while True:
    texto = input("> ")
    if texto.lower() == "sair":
        print("Encerrando...")
        break
    
    X_teste = vectorizer.transform([texto])
    pred = model.predict(X_teste)[0]
    
    if str(pred) == "1":
        resultado = "Negativo"
    else:
        resultado = "Positivo"
    
    print(f"Classificação: {resultado}\n")
