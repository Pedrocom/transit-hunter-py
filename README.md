# Lightcurve Exoplanet Pipeline

Pipeline reprodutível em Python para baixar, limpar, analisar e visualizar curvas de luz astronômicas (Kepler / TESS) usando `lightkurve` e busca de trânsitos via Box Least Squares (BLS).

## Estrutura do projeto

- `src/fetch.py`: baixa a curva de luz e salva em `.parquet` em `data/processed/`.
- `src/plot_lc.py`: gera visualização inicial da curva normalizada em `outputs/`.
- `src/analyze_bls.py`: aplica *detrending*, roda busca BLS e salva gráficos da curva achatada e dobrada em fase.

## Ambiente e execução

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\fetch.py
python src\plot_lc.py
python src\analyze_bls.py