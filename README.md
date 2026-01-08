# Project-Work-2026
This is a Predictive Analysis of United Kingdom Good Exports Post-Brexit
# Predictive Analysis of UK Goods Exports Post‑Brexit

This repository contains the code and artefacts for a forecasting framework that compares traditional econometric models with machine‑learning methods to predict UK goods exports to the EU in the post‑Brexit period. The core workflow ingests trade and macroeconomic data, trains SARIMAX, Prophet, and XGBoost models, builds a weighted ensemble, and generates both quantitative evaluation metrics and visualisations.[1]

## Repository structure

- `Aretefact.ipynb`  
  End‑to‑end notebook that:  
  - Loads and preprocesses trade and macroeconomic data  
  - Trains SARIMAX, Prophet, and XGBoost models  
  - Builds a weighted ensemble based on inverse MAE  
  - Saves forecasts, metrics, and feature importance to CSV  
  - Describes the separate visualisation script used to create charts and tables[1]

- `analysis_and_modeling.py` (described inside the notebook)  
  Script version of the modelling pipeline (Phases 1–6).[1]

- `visualization.py` (described inside the notebook)  
  Script that loads the saved CSVs and produces publication‑ready charts and a metrics table.[1]

- Output files (created after running the pipeline):  
  - `forecasts_and_results.csv` – actual values, individual model forecasts, and ensemble forecasts  
  - `evaluation_metrics.csv` – extended metrics for each model (RMSE, MAPE, MASE, CDC, Theil’s U, etc.)  
  - `xgb_feature_importance.csv` – XGBoost feature importance scores[1]

## Data and features

The pipeline expects monthly UK trade data and a policy‑uncertainty proxy:[1]

- Trade data: exports to EU and non‑EU markets, 2019‑01 to 2024‑12  
- Macroeconomic indicator: `SERI_Proxy` (Sterling Effective Rate Index )  

Key engineered features for XGBoost include:

- Lagged exports: \(L1, L2, L3, L6, L12\)  
- Rolling statistics: 3‑month rolling mean and standard deviation  
- Brexit indicator: binary flag set to 1 from 2021‑01‑01 onwards  
- Lagged macro variables: `SERI_Proxy_lag1`, `SERI_Proxy_lag2`  
- Calendar features: year, month, quarter[1]

If the CSV files are missing, the loader automatically generates mock data with the same structure so the pipeline can still be executed.[1]

## Modelling workflow

1. **Data ingestion & preprocessing**  
   - Load trade and macro CSVs (`monthly_exports2019-20251.csv`, `mret new.csv`).  
   - Reshape country‑level data to aggregated EU vs Non‑EU monthly exports.  
   - Merge with `SERI_Proxy` and split into train (2019‑01–2022‑12) and test (2023‑01–2024‑12).[1]

2. **Extended metric evaluation**  
   - `calculate_extended_metrics()` computes:  
     - RMSE, MAE, MAPE, SMAPE  
     - MASE (against a seasonal naïve benchmark)  
     - R², RMSLE  
     - Theil’s U  
     - Change Directional Accuracy (CDC/DAC)[1]

3. **Model training**  
   - **SARIMAX**  
     - Orders selected via `pmdarima.auto_arima` with `SERI_Proxy` as exogenous regressor.  
   - **Prophet**  
     - Yearly seasonality, custom changepoint prior, and `SERI_Proxy` as extra regressor.  
   - **XGBoost**  
     - Gradient‑boosting regressor trained on engineered feature set.  
     - StandardScaler applied to non‑binary features.  
     - TimeSeriesSplit walk‑forward cross‑validation with GridSearchCV for hyperparameter tuning.[1]

4. **Ensemble creation**  
   - Compute inverse‑MAE weights for SARIMAX, Prophet, and XGBoost on the test set.  
   - Form a weighted ensemble forecast as the MAE‑weighted average of individual model predictions.[1]

5. **Outputs & visualisation**  
   - Save combined actuals/forecasts, evaluation metrics, and feature importance.  
   - `visualization.py` produces:  
     - Forecast vs actuals plot with train/test shading and Brexit break line  
     - EU vs Non‑EU export trend comparison  
     - XGBoost top‑10 feature importance bar chart  
     - Markdown‑style metrics table highlighting the best model by MAE[1]

## How to run

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Typical dependencies (as used in the notebook): `pandas`, `numpy`, `statsmodels`, `prophet`, `pmdarima`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`.[1]

4. **Place data files**

Ensure the following files are in the project root (or update paths in the code):

- `monthly_exports2019-20251.csv`  
- `mret new.csv`[1]

5. **Run the modelling pipeline**

- Either execute all cells in `Aretefact.ipynb` in order, or  
- Run the script version:

```bash
python analysis_and_modeling.py
```

This will generate `forecasts_and_results.csv`, `evaluation_metrics.csv`, and `xgb_feature_importance.csv`.[1]

6. **Generate visualisations**

```bash
python visualization.py
```

This will display plots and print the final metrics summary, including the best model based on MAE.[1]

## Reproducibility notes

- The time split (train: 2019‑01–2022‑12, test: 2023‑01–2024‑12) is fixed for comparability.  
- For XGBoost, a fixed random seed is used to stabilise results.  
- If real data is unavailable, the pipeline falls back to mock data with the same schema to demonstrate the full workflow.[1]

***

You can tweak project title, repo URL, and the “Data and features” section if your CSV filenames or indicator names differ.
