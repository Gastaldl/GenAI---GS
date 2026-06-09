# Previsão de Desmatamento via Dados de Satélite

**FIAP · Global Solution 2026 · Indústria Espacial**
**Disciplina:** Generative AI for Engineering (GAIE)

Pipeline completo de **IA/ML tabular** que usa indicadores ambientais derivados de
**observação da Terra por satélite** (World Bank · Landsat/MODIS/Sentinel) para **prever a
variação da cobertura florestal do próximo ciclo** (regressão) e **classificar o risco de
desmatamento** de cada país/região (classificação), com **interpretabilidade via SHAP** e
**deploy** em um dashboard interativo.

> Conexão com a Indústria Espacial: os dados de cobertura florestal e uso do solo do World
> Bank são produzidos a partir de sensoriamento remoto por satélite. Este projeto é a
> **camada analítica de um produto de dados espaciais** — transforma observação da Terra em
> previsão acionável. **ODS prioritário: ODS 13 (Ação Climática)** (conexões com ODS 2 e 11).

---

## Integrantes


| Nome completo | RM |
| ------------- | --------- |
| Márcio Gastaldi | RM98811 |
| Fabrício Gutierrez Saavedra | RM97631 |
| Arthur Bessa Pian | RM99215 |
| Davi Desenzi | RM550849 |
| João Victor | RM551410 |


---

## Links da entrega

- **Repositório GitHub:** https://github.com/Gastaldl/GenAI---GS.git
- **Aplicação em funcionamento (Hugging Face Spaces):** https://huggingface.co/spaces/Gastaldl/gs-desmatamento-satelite

---

## 1. Contexto do problema

O desmatamento é um dos maiores vetores de emissão de carbono e perda de biodiversidade.
Monitorar e **antecipar** a perda de cobertura florestal permite que governos e ONGs ajam
preventivamente. Hoje, dados de satélite cobrem o planeta inteiro — mas dados sozinhos não
decidem nada. É preciso uma **camada de IA** que transforme indicadores ambientais em uma
**previsão quantitativa de risco**.

**Quem sofre:** populações de regiões em desmatamento acelerado, o clima global e o
agronegócio que depende de equilíbrio hídrico/florestal.
**Por que importa:** cada ponto percentual de floresta perdido tem custo climático e econômico
mensurável; prever a tendência habilita políticas públicas e alocação de fiscalização.

## 2. Fonte dos dados

- **Primária (real, gratuita, sem chave):** [World Bank Indicators API](https://data.worldbank.org/).
Painel `país × ano` (2000–2022, 217 países/regiões), com **12 indicadores** ambientais e
socioeconômicos (cobertura florestal, terra agrícola/arável, população urbana, densidade,
PIB per capita, CO₂ per capita, energia renovável, poluição PM2.5, área de terra etc.).
- **Apenas dados reais:** usamos a opção *API real* do enunciado. Se a API estiver indisponível,
`src/data_loader.py` levanta um **erro claro** (sem gerar dados falsos), preservando a
integridade do projeto. Os CSVs já baixados ficam versionados em `data/`, permitindo rodar o
restante do pipeline (pré-processamento, treino e app) **offline**.

Dataset resultante: **4.991 linhas × 17 colunas** → após engenharia de atributos: **36 colunas**
(31 features + alvos), com **4.686 linhas modeláveis**.

## 3. Definição do problema de ML

A partir do estado de um país no ano *t*, prevemos o que acontece com a floresta em *t+1*:

- **Regressão (`target_reg`)**: variação **percentual** da área de floresta no ano seguinte (% a.a.),
limitada a [-25%, +25%] para conter outliers.
- **Classificação (`target_clf`)**: classe de risco derivada da variação:
  - `Desmatamento Alto` (queda > 0,30% a.a.)
  - `Estável`
  - `Recuperação` (ganho > 0,30% a.a.)

Distribuição das classes (modeláveis): Estável 2.723 · Desmatamento Alto 1.108 · Recuperação 855.

**Sem vazamento temporal:** todas as features são do ano *t* (ou anteriores, via *lags*); os
alvos olham para *t+1*. A validação usa **split temporal** + **GroupKFold por país**.

## 4. Metodologia (pipeline)

```
World Bank API (dados reais)             src/data_loader.py
        └─► data/raw/worldbank_panel.csv
              └─► src/preprocessing.py  (imputação + engenharia de atributos + alvos)
                    └─► data/processed/dataset.csv
                          └─► src/train.py  (treina/compara modelos, valida, salva melhores)
                                ├─► models/*.joblib
                                ├─► reports/*.csv  +  reports/figures/*.png
                                └─► src/explain.py (SHAP global/local)
                                      └─► app/app.py (Streamlit) ─► Hugging Face Spaces
```

- **Pré-processamento:** remoção de agregados regionais; imputação por **interpolação temporal
por país** + mediana global; padronização (`StandardScaler`) dos numéricos e `OneHotEncoder`
dos categóricos via `ColumnTransformer`/`Pipeline`.
- **Engenharia de atributos:** *lags* (ano anterior), variações ano-a-ano (YoY), média móvel
de 3 anos da cobertura florestal e razões de pressão (agrícola/florestal, pressão urbana,
índice de desenvolvimento). Total: **29 features numéricas + 2 categóricas**.
- **Modelos comparados (≥2 técnicas):**
  - Regressão: `LinearRegression`, `RandomForest`, `GradientBoosting`, `XGBoost`.
  - Classificação: `LogisticRegression`, `RandomForest`, `GradientBoosting`, `XGBoost`.
- **Validação:** `GroupKFold` (5 folds, agrupado por país) no treino + métricas em conjunto de
**teste temporal** (anos > 2016). Métricas: MAE, RMSE, R² (regressão); acurácia, F1-macro,
ROC-AUC (OvR) e matriz de confusão (classificação).

## 5. Modelos testados e resultados

### Regressão (variação % da floresta no ano seguinte)


| Modelo                    | CV R² (treino) | Test R²   | Test MAE  | Test RMSE |
| ------------------------- | -------------- | --------- | --------- | --------- |
| **RandomForest** (melhor) | 0.389          | **0.758** | **0.131** | **0.373** |
| GradientBoosting          | 0.294          | 0.734     | 0.193     | 0.391     |
| XGBoost                   | 0.306          | 0.715     | 0.157     | 0.405     |
| LinearRegression          | 0.049          | −0.189    | 0.356     | 0.827     |


> O modelo linear falha (R² negativo): a relação é **não-linear**, o que justifica os modelos
> baseados em árvores. **Melhor regressor: RandomForest.**

### Classificação (risco de desmatamento)


| Modelo               | CV F1-macro | Test Acc  | Test F1-macro | Test ROC-AUC (OvR) |
| -------------------- | ----------- | --------- | ------------- | ------------------ |
| **XGBoost** (melhor) | 0.801       | **0.966** | **0.956**     | 0.989              |
| GradientBoosting     | 0.816       | 0.956     | 0.943         | 0.989              |
| RandomForest         | 0.712       | 0.954     | 0.939         | 0.986              |
| LogisticRegression   | 0.748       | 0.869     | 0.808         | 0.952              |


Relatório do melhor classificador (XGBoost, teste): `Desmatamento Alto` F1=0.97 · `Estável`
F1=0.97 · `Recuperação` F1=0.93 — **acurácia 0.97**. **Melhor classificador: XGBoost.**

Figuras em `[reports/figures/](reports/figures/)`: comparação de modelos, matriz de confusão,
dispersão previsto×real e gráficos SHAP.

## 6. Interpretabilidade com SHAP

Usamos `shap.TreeExplainer` sobre o melhor modelo de árvore:

- **Global:** `reports/figures/shap_regression_beeswarm.png` e `shap_regression_bar.png`
mostram que a **tendência recente da floresta (YoY)**, a **cobertura atual** e a **média móvel**
são os fatores mais influentes.
- **Local:** o app exibe um **waterfall** explicando *cada previsão* — por que aquela região
recebeu determinada projeção/risco.

## 7. Como executar (do zero)

> Pré-requisitos: **Python 3.11+** e **Git**. Testado em Python 3.14 (Windows).

```powershell
# 1) Clonar o repositório
git clone https://github.com/Gastaldl/GenAI---GS.git
cd GenAI---GS

# 2) Ambiente virtual
py -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate             # Linux/Mac

# 3) Dependências
pip install -r requirements.txt

# 4) Pipeline completo
python src/data_loader.py     # baixa dados reais do World Bank
python src/preprocessing.py   # limpeza + engenharia de atributos + alvos
python src/train.py           # treina/compara modelos e salva os melhores em models/
python src/explain.py         # gera as figuras SHAP globais

# 5) Aplicação (dashboard)
streamlit run app/app.py      # abre em http://localhost:8501
```

> No Windows, se `python` abrir a Microsoft Store, use `py` no lugar de `python`.

## 8. Estrutura do repositório

```
README.md                      # este arquivo
requirements.txt               # dependências (dev/local)
notebooks/pipeline_gaie.ipynb  # notebook end-to-end (EDA → treino → SHAP), executado
src/
  data_loader.py               # World Bank API (dados reais)
  preprocessing.py             # limpeza, imputação, engenharia de atributos, alvos
  train.py                     # treino/comparação de modelos + validação
  evaluate.py                  # métricas e gráficos
  explain.py                   # SHAP (global e local)
app/app.py                     # dashboard Streamlit (entry do HF Space)
models/                        # best_regressor / best_classifier / metadata (.joblib)
data/raw/, data/processed/     # painel bruto e dataset processado
reports/, reports/figures/     # tabelas de métricas e figuras
space/                         # arquivos para o Hugging Face Spaces
```

## 9. Reprodutibilidade

- Seeds fixas (`random_state=42`) em todos os modelos.
- Versões congeladas em `[requirements-lock.txt](requirements-lock.txt)` (geradas via `pip freeze`).
- O dataset processado e os modelos treinados são versionados para permitir rodar o app sem
re-treinar.

---

*FIAP Global Solution 2026 · Indústria Espacial · GAIE · ODS 13*
