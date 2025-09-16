# app/app.py
import pickle
from pathlib import Path
import streamlit as st
from sklearn.pipeline import Pipeline  # usado no fallback

# --- caminhos dos artefatos ---
ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

PIPE_PATH = NB_DIR / "pipeline.pkl"
MODEL_PATH = NB_DIR / "modelo_treinado.pkl"
VEC_PATH  = NB_DIR / "vectorizer.pkl"

@st.cache_resource
def load_pipeline():
    # 1) tentar pipeline.pkl
    if PIPE_PATH.exists():
        with open(PIPE_PATH, "rb") as f:
            pipe = pickle.load(f)
        return pipe, f"Carregado: {PIPE_PATH.name}"

    # 2) fallback: reconstruir a partir de vectorizer + modelo
    if MODEL_PATH.exists() and VEC_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            clf = pickle.load(f)
        with open(VEC_PATH, "rb") as f:
            vec = pickle.load(f)
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        # opcional: salvar para próximas execuções
        try:
            NB_DIR.mkdir(parents=True, exist_ok=True)
            with open(PIPE_PATH, "wb") as f:
                pickle.dump(pipe, f)
            return pipe, f"Reconstruído de {VEC_PATH.name} + {MODEL_PATH.name} → salvo como {PIPE_PATH.name}"
        except Exception:
            return pipe, f"Reconstruído de {VEC_PATH.name} + {MODEL_PATH.name}"

    # 3) nada encontrado
    raise FileNotFoundError(
        "Artefatos não encontrados. Coloque em notebooks/ um destes conjuntos:\n"
        f" - {PIPE_PATH.name}\n"
        f" - {MODEL_PATH.name} e {VEC_PATH.name}"
    )

# --- UI ---
st.set_page_config(page_title="Analisador de Sentimentos", page_icon="🔎")
st.title("🔎 Analisador de Sentimentos")

pipeline, origem = load_pipeline()
st.caption(f"Artefatos: {origem}")

texto = st.text_area("Digite uma avaliação de produto:")
clicked = st.button("Classificar")  # >>> um único botão <<<

if clicked:
    if not texto.strip():
        st.warning("Digite um texto antes de classificar.")
    else:
        pred = pipeline.predict([texto])[0]        # mapeamento: 1=Negativo, 2=Positivo
        # pode não existir em todos os modelos, por isso try
        try:
            proba = pipeline.predict_proba([texto])[0]  # [P(1), P(2)]
        except Exception:
            proba = None

        if int(pred) == 1:
            if proba is not None:
                st.error(f"Resultado: **Negativo** ❌ — conf.: {proba[0]:.2%}")
            else:
                st.error("Resultado: **Negativo** ❌")
        else:
            if proba is not None:
                st.success(f"Resultado: **Positivo** ✅ — conf.: {proba[1]:.2%}")
            else:
                st.success("Resultado: **Positivo** ✅")

        with st.expander("🔧 Detalhes"):
            st.write({"pred": int(pred), "probas": None if proba is None else proba.tolist()})
