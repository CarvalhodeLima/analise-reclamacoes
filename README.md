Análise de Reclamações sobre Produtos Eletrônicos

Contexto
As avaliações online exercem grande influência nas decisões de compra de consumidores. Produtos com muitas reclamações podem impactar negativamente a reputação das marcas e reduzir suas vendas.

Este projeto utiliza o dataset Amazon Electronics Reviews como base para identificar e analisar os principais motivos de insatisfação dos clientes em produtos eletrônicos.

Objetivo
- Verificar quais produtos apresentam mais reclamações
- Identificar os principais motivos de insatisfação relatados pelos clientes
- Construir um modelo capaz de analisar automaticamente novas avaliações
- Fornecer uma interface interativa em Streamlit para consulta dos resultados

Tecnologias Utilizadas
- Python 3.10+
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
- NLTK
- Streamlit
- SQLite


Como executar
1. Ativar o ambiente virtual:
   .\venv\Scripts\Activate.ps1   # PowerShell
   venv\Scripts\activate.bat     # CMD

2. Instalar as dependências:
   pip install -r requirements.txt

3. Rodar a aplicação no Streamlit:
   streamlit run app/app.py

Métricas de Avaliação
- RMSE para modelos de regressão
- Matriz de Confusão para modelos de classificação

Documentação
Toda a documentação adicional está localizada na pasta docs/, incluindo:
- pmc.md 
- architecture.md → Arquitetura do sistema
- datamodel.md → Modelagem de dados (SOR, SOT, SPEC)
