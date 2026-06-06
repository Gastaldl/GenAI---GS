# Guia de Deploy — Hugging Face Spaces (Streamlit)

Este guia leva o app ao ar com uma **URL pública gratuita**. Faça os passos
manualmente (não precisa de cartão de crédito).

## Pré-requisito: modelos treinados
Antes de subir, rode o pipeline localmente para gerar `models/` e `data/processed/dataset.csv`:

```powershell
python src/data_loader.py
python src/preprocessing.py
python src/train.py
```

## Passo 1 — Criar conta e Space
1. Crie uma conta em https://huggingface.co (e confirme o e-mail).
2. Vá em **New** → **Space**.
3. Preencha:
   - **Owner:** seu usuário
   - **Space name:** escolha um nome (ex.: `gs-desmatamento-satelite`)
   - **License:** MIT
   - **SDK:** **Streamlit**
   - **Hardware:** CPU basic (gratuito)
   - **Visibility:** Public
4. Clique em **Create Space**.

## Passo 2 — Enviar os arquivos
Você pode usar a **interface web** (mais fácil) ou **git**.

### Opção A — Interface web (recomendada)
Na aba **Files** do Space, clique em **Add file → Upload files** e envie, mantendo a estrutura:

| Origem (neste repo) | Destino no Space |
|---|---|
| `app/app.py` | `app.py` (na **raiz**) |
| `space/requirements.txt` | `requirements.txt` (na **raiz**) |
| `space/README.md` | `README.md` (na **raiz**, com o cabeçalho YAML) |
| `models/best_regressor.joblib` | `models/best_regressor.joblib` |
| `models/best_classifier.joblib` | `models/best_classifier.joblib` |
| `models/metadata.joblib` | `models/metadata.joblib` |
| `data/processed/dataset.csv` | `data/processed/dataset.csv` |

> Para criar subpastas no upload web, digite o caminho no nome do arquivo (ex.:
> `models/best_regressor.joblib`).

### Opção B — Git
```bash
git clone https://huggingface.co/spaces/<seu-usuario>/<nome-do-space>
cd <nome-do-space>
# copie para ca: app.py (de app/app.py), requirements.txt e README.md (de space/),
# a pasta models/ e data/processed/dataset.csv
git add .
git commit -m "Deploy do app GAIE"
git push
```
> O upload de `.joblib` pode exigir **Git LFS** (`git lfs install` e `git lfs track "*.joblib"`).
> Pela interface web não é necessário.

## Passo 3 — Build e teste
- O Space inicia o **build** automaticamente (aba **Logs**). Leva ~2–5 min na primeira vez.
- Quando aparecer **Running**, o app estará disponível na URL pública:
  `https://huggingface.co/spaces/<seu-usuario>/<nome-do-space>`
- Teste: selecione um país, ajuste cenários e confira a previsão + risco + SHAP.

## Passo 4 — Registrar o link
Cole a URL pública na seção **Links da entrega** do `README.md` do GitHub e no portal FIAP.

## Solução de problemas
- **Erro ao carregar modelos / pickle:** confirme que o `requirements.txt` do Space usa as
  **mesmas versões** do treino (este repo já as fixa) e que `python_version: "3.12"` está no
  cabeçalho do `README.md`.
- **App não encontra dados:** verifique se `data/processed/dataset.csv` e a pasta `models/`
  foram enviados com os caminhos corretos.
- **Build lento/timeout:** reinicie o Space em **Settings → Factory reboot**.
