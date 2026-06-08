"""
Global configuration for the StatArb pipeline.
Override any value via environment variables or a local .env file.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class DataConfig(BaseModel):
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    cache_dir: Path = ROOT / "data" / "cache"
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    interval: str = "1d"
    alpha_vantage_key: str = Field(default_factory=lambda: os.getenv("AV_API_KEY", "demo"))


class AnalysisConfig(BaseModel):
    correlation_window: int = 252          # rolling window in trading days
    correlation_min: float = 0.70          # minimum absolute Pearson for screening
    coint_pvalue_max: float = 0.05         # Engle-Granger p-value threshold
    adf_pvalue_max: float = 0.05           # ADF p-value on spread
    hurst_max: float = 0.50                # Hurst exponent upper bound (mean-reverting)
    half_life_min: int = 5                 # days
    half_life_max: int = 60               # days


class MLConfig(BaseModel):
    test_size: float = 0.20
    cv_folds: int = 5
    n_estimators: int = 500
    random_state: int = 42
    optuna_trials: int = 50
    lstm_lookback: int = 60
    lstm_epochs: int = 30
    lstm_batch_size: int = 32
    top_n_pairs: int = 20


class StrategyConfig(BaseModel):
    zscore_entry: float = 2.0              # enter when |z| crosses this
    zscore_exit: float = 0.5              # exit when |z| reverts here
    zscore_stop: float = 3.5              # hard stop-loss z-score
    lookback_zscore: int = 60             # rolling window for z-score
    kalman_transition_cov: float = 1e-4
    kalman_observation_cov: float = 1e-2
    max_holding_days: int = 30


class BacktestConfig(BaseModel):
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001         # 0.10 % per leg
    slippage_pct: float = 0.0005
    risk_free_rate: float = 0.05          # annualised


class SimulationConfig(BaseModel):
    n_paths: int = 10_000
    horizon_days: int = 252
    confidence_levels: list[float] = [0.95, 0.99]
    random_seed: int = 42


class OutputConfig(BaseModel):
    reports_dir: Path = ROOT / "outputs" / "reports"
    charts_dir: Path = ROOT / "outputs" / "charts"
    log_level: str = "INFO"


class Settings(BaseModel):
    data: DataConfig = DataConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    ml: MLConfig = MLConfig()
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()
    simulation: SimulationConfig = SimulationConfig()
    output: OutputConfig = OutputConfig()


# Singleton instance used across the project
settings = Settings()

# Ensure output directories exist
for _dir in [
    settings.data.raw_dir,
    settings.data.processed_dir,
    settings.data.cache_dir,
    settings.output.reports_dir,
    settings.output.charts_dir,
]:
    _dir.mkdir(parents=True, exist_ok=True)
