# StatArb — Statistical Arbitrage Pipeline

A production-ready, end-to-end statistical arbitrage system combining
correlation/cointegration analysis, ML pair selection, vectorised backtesting,
and Monte Carlo simulation — served through an interactive Streamlit dashboard.

---

## Architecture

```
Data Collection → Preprocessing → Correlation → Cointegration
        ↓
Feature Engineering → ML Selection (XGBoost + LSTM)
        ↓
Signal Generation (Z-score / Kalman Filter)
        ↓
Backtesting (vectorised) + Benchmark Comparison
        ↓
Monte Carlo Simulation + Stress Tests
        ↓
Streamlit Dashboard + PDF Report
```

---

## Project Structure

```
statarb/
├── config/
│   ├── settings.py        # Pydantic global config (thresholds, paths, params)
│   └── universe.yaml      # Watchlist: equities, ETFs, commodities by sector
│
├── src/
│   ├── data/
│   │   ├── collectors.py  # yfinance + Alpha Vantage downloaders (joblib cached)
│   │   ├── preprocessors.py  # Cleaning, returns, winsorisation, z-score
│   │   └── universe.py    # Universe loading, liquidity filter, pair generation
│   │
│   ├── analysis/
│   │   ├── correlation.py  # Pearson/Spearman, rolling, cluster-based screening
│   │   ├── cointegration.py  # Engle-Granger, Johansen, ADF, KPSS, half-life
│   │   └── spread.py       # OLS/rolling-OLS/Kalman spread, Hurst, OU params
│   │
│   ├── features/
│   │   └── builder.py      # Feature matrix for ML (20 features per pair)
│   │
│   ├── models/
│   │   ├── ml_selector.py  # XGBoost / Random Forest + Optuna + SHAP
│   │   ├── lstm_selector.py  # Regime-aware LSTM (PyTorch)
│   │   └── pair_ranker.py  # Composite scoring and top-N selection
│   │
│   ├── strategy/
│   │   ├── signals.py      # Z-score state machine, Kalman-filter signals
│   │   ├── execution.py    # Commission, slippage, PnL, portfolio aggregation
│   │   └── risk.py         # Kelly/vol-target sizing, drawdown halt, stop-loss
│   │
│   ├── backtest/
│   │   ├── engine.py       # Full backtest orchestrator (walk-forward ready)
│   │   ├── metrics.py      # Sharpe, Sortino, Calmar, VaR, CVaR, trade stats
│   │   └── benchmarks.py   # Buy-and-hold, momentum, mean-reversion, 60/40
│   │
│   └── simulation/
│       ├── monte_carlo.py  # GBM + block bootstrap, 10k paths, fan chart
│       └── scenario.py     # Vol shock, correlation breakdown, liquidity crisis
│
├── dashboard/
│   └── app.py              # Streamlit UI — 6 tabs, full Plotly charts
│
├── tests/
│   ├── test_cointegration.py
│   ├── test_signals.py
│   └── test_backtest.py
│
├── main.py                 # CLI entry point
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **VPS / CPU-only note**: PyTorch CPU build is the default in `requirements.txt`.
> The LSTM module is optional — the pipeline runs fully without it.

### 2. Configure

Edit `config/universe.yaml` to customise your watchlist.
Set your Alpha Vantage key if needed:

```bash
export AV_API_KEY=your_key_here
```

Adjust thresholds in `config/settings.py` (all Pydantic-validated).

### 3. Run the full pipeline (CLI)

```bash
# End-to-end: collect → analyze → train → backtest → simulate
python main.py --mode full

# Individual steps
python main.py --mode collect
python main.py --mode analyze
python main.py --mode train
python main.py --mode backtest
python main.py --mode simulate
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
# or
python main.py --mode dashboard
```

Navigate to `http://localhost:8501`. Click **▶ Run Full Pipeline** in the sidebar.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Key parameters (config/settings.py)

| Parameter | Default | Description |
|---|---|---|
| `analysis.correlation_min` | 0.70 | Min \|Pearson r\| to screen pairs |
| `analysis.coint_pvalue_max` | 0.05 | EG p-value threshold |
| `analysis.half_life_min/max` | 5 / 60 | Acceptable mean-reversion half-life (days) |
| `strategy.zscore_entry` | 2.0 | Z-score to open a position |
| `strategy.zscore_exit` | 0.5 | Z-score to close a position |
| `strategy.zscore_stop` | 3.5 | Hard stop-loss z-score |
| `ml.top_n_pairs` | 20 | Pairs selected for trading |
| `backtest.commission_pct` | 0.001 | Commission per leg (0.1 %) |
| `simulation.n_paths` | 10,000 | Monte Carlo paths |

---

## Dashboard tabs

| Tab | Contents |
|---|---|
| 📦 Universe & Data | Ticker table, sector breakdown, rebased price chart |
| 🔗 Correlation | Full heatmap, high-corr pair table, rolling correlation explorer |
| ⚗️ Cointegration & ML | Results table, half-life scatter, Kalman spread explorer, ML scores |
| 📊 Backtest | Equity curve, drawdown, rolling Sharpe, per-pair metrics |
| 🎲 Monte Carlo | GBM & bootstrap fan charts, VaR/CVaR, stress-test scenario table |
| ⚖️ Comparison | Metrics table, equity curve overlay, risk-return radar |

---

## Design notes

**Pair screening order** (each step reduces the candidate set):
1. Same-sector filter (economic rationale)
2. Correlation screen `|r| >= 0.70`
3. Engle-Granger + ADF cointegration tests
4. Half-life filter `5d ≤ HL ≤ 60d`
5. Hurst exponent filter `H < 0.5`
6. ML composite score → top-N

**Spread construction** (three options):
- `ols` — static OLS hedge ratio, fastest
- `rolling_ols` — 60-day rolling OLS, adapts slowly
- `kalman` — Kalman filter, fully dynamic (default, recommended)

**Walk-forward testing**: pass `train_end="2021-12-31"` to `BacktestEngine.run()`
to restrict trading to the out-of-sample period only.
