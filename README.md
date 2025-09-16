# Análise de Reclamações — Amazon Electronics (Sentimento)

Classificador binário (negativo/positivo) de reviews de produtos eletrônicos. Treino feito sobre dados em inglês (formato fastText).

ATENÇÃO SOBRE IDIOMA
- O modelo foi treinado em inglês. Para melhor qualidade, teste com frases em inglês.

## Requisitos
- Python 3.10+ (recomendado 3.12)
- Pip/venv
- Dependências em `requirements.txt`

## Setup do ambiente
Windows:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt


## Como executar o app
streamlit run app/app.py


O app carrega artefatos de `notebooks/`:
- pipeline.pkl  (recomendado)
ou
- modelo_treinado.pkl + vectorizer.pkl  (o app reconstrói um pipeline na inicialização)

## Métricas (para relatório)
- docs/classification_report.txt
- docs/confusion_matrix.png
Obs.: RMSE não se aplica a classificação. Use Accuracy, Precision, Recall, F1 e Matriz de Confusão.

## Estrutura do projeto (resumo)
- app/                 → Streamlit (predição)
- core/data/           → leitura, banco (SQLite utilitário), SQL (SOR/SOT/SPEC)
- core/features/       → preprocessamento
- notebooks/           → treino/experimentos + artefatos (.pkl)
- docs/                → arquitetura, canvas, métricas
- data/                → (opcional) amostras pequenas só para demonstração de treino local


