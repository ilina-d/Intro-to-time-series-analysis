# Intro to Time Series Analysis

A collection of Jupyter notebooks exploring time series analysis and forecasting, from elementary univariate analysis to comparing a wide range of forecasting models.

[Notebooks](#notebooks) • [Topics Covered](#topics-covered) • [Implemented Models](#implemented-models) • [Usage](#usage)

---

## Notebooks

| Notebook | Dataset | Description |
|---|---|---|
| [`elementary_analysis.ipynb`](https://github.com/ilina-d/Intro-to-time-series-analysis/blob/main/elementary_analysis.ipynb) | Egg sales of a local shop in Sri Lanka (30 years) | Elementary analysis of a univariate time series. |
| [`univariate_vs_multivariate.ipynb`](https://github.com/ilina-d/Intro-to-time-series-analysis/blob/main/univariate_vs_multivariate.ipynb) | Renewable power generation & weather conditions (2017–2021, 15-min intervals) | Forecasting GHI using multiple univariate and multivariate models. |

---

## Topics Covered

### 1. Elementary Analysis (`elementary_analysis.ipynb`)

- **Visualization** — plotting the full series and sub-segments (last 10, 5, 3 years) to identify structure
- **Decomposition** — additive and multiplicative decomposition using `seasonal_decompose`; linear interpolation of COVID-era zero values before decomposing
- **Stationarity testing** — Augmented Dickey-Fuller (ADF) test on the full series and on sub-segments
- **Detrending** — two approaches: subtracting the decomposition trend component, and subtracting the least-squares line of best fit
- **Deseasonalization** — confirming yearly seasonality with a monthly heatmap and ANOVA F-test, then removing the seasonal component
- **Autocorrelation & Partial Autocorrelation** — ACF/PACF plots across 400 lags, and on detrended data to highlight lag structure
- **Smoothing** — residual removal via decomposition, and rolling average smoothing with 7-day and 30-day windows

### 2. Univariate vs. Multivariate Forecasting (`univariate_vs_multivariate.ipynb`)

- **Pre-processing** — resampling to a full 15-minute grid, forward-fill for missing values, trimming sparse boundary periods
- **Visualization & seasonality checks** — GHI heatmaps across hour/day/month/year/season dimensions
- **ACF/PACF** — daily and yearly autocorrelation analysis, including daily-aggregated data for long-range lags
- **Train/test split** — 80/20 chronological split
- **Rolling window and sampled-section forecasting strategies** — applied consistently across all models for fair comparison

---

## Implemented Models

### Univariate

| Model | Description |
|---|---|
| FB Prophet | Additive model with configurable daily and yearly seasonality; both fixed and rolling window strategies tested. |
| SARIMA | Seasonal ARIMA with order `(1,0,0)(1,1,0)[96]`; sampled-section rolling window predictions (full rolling too slow on 15-min data). |

### Multivariate

| Model | Description |
|---|---|
| VAR | Vector Autoregression; tested with `[GHI, isSun]` and `[GHI, temp, humidity, isSun]`; lag order selected via AIC. |
| LightGBM | Gradient boosting with lag features and rolling means; tested in univariate and multivariate (`temp`, `humidity`) configurations. |
| Ridge Regression | Linear model with lag/rolling features and `StandardScaler`; multivariate with `temp`, `humidity`, `isSun`. |
| Lasso Regression | Same setup as Ridge, applied for comparison. |
| LSTM (pretrained + finetuned) | Two-layer LSTM pretrained on the training set and finetuned per section; Huber loss with cosine LR schedule and early stopping. |

---

## Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet pmdarima \
            lightgbm scikit-learn torch optuna missingno kagglehub jupyter
```

### Running the Notebooks

```bash
# Clone the repository
git clone https://github.com/ilina-d/Intro-to-time-series-analysis.git
cd Intro-to-time-series-analysis

# Launch Jupyter
jupyter notebook
```

Then open any `.ipynb` file from the Jupyter interface in your browser. Both notebooks download their datasets automatically via `kagglehub`.
