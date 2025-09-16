# Análise de Reclamações (Produtos Eletrônicos)

App de classificação de sentimento (positivo/negativo) de reviews (Amazon Electronics).

## Como rodar
1) Python 3.12 + venv
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
2) Streamlit
   streamlit run app/app.py

Obs.: Modelo treinado em inglês. Use frases em EN (ou ative a tradução no app, se disponível).

## Artefatos
- notebooks/pipeline.pkl  (ou modelo_treinado.pkl + vectorizer.pkl)

## Métricas
- docs/classification_report.txt
- docs/confusion_matrix.png

## Estrutura
app/  core/  notebooks/  docs/  data/

## Nota
Dataset bruto não incluso por tamanho; o app usa apenas os artefatos .pkl.
