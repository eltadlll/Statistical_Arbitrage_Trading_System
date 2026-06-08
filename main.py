"""
main.py
-------
CLI entry point for the Statistical Arbitrage pipeline.

Usage:
  python main.py --mode collect
  python main.py --mode analyze
  python main.py --mode train
  python main.py --mode backtest
  python main.py --mode simulate
  python main.py --mode full          # end-to-end

  python main.py --mode dashboard     # launch Streamlit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.data.collectors import DataCollector
from src.data.preprocessors import Preprocessor
from src.data.universe import UniverseBuilder
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.features.builder import FeatureBuilder
from src.models.ml_selector import MLPairSelector
from src.models.pair_ranker import PairRanker
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics
from src.backtest.benchmarks import BenchmarkRunner
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.scenario import ScenarioTester

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logger setup
# ─────────────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.output.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)
logger.add(
    settings.output.reports_dir / "pipeline.log",
    rotation="10 MB",
    level="DEBUG",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps (each callable independently)
# ─────────────────────────────────────────────────────────────────────────────

def step_collect() -> pd.DataFrame:
    logger.info("═══ STEP: Data Collection ═══")
    ub = UniverseBuilder()
    ub.load()
    tickers = ub.all_tickers()
    collector = DataCollector()
    prices = collector.fetch(tickers)
    collector.save_raw(prices, "prices")
    logger.success(f"Saved {prices.shape[1]} tickers × {prices.shape[0]} days")
    return prices


def step_preprocess(prices: pd.DataFrame) -> dict:
    logger.info("═══ STEP: Preprocessing ═══")
    pp = Preprocessor()
    data = pp.run(prices)

    for name, df in data.items():
        path = settings.data.processed_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info(f"  Saved {name} → {path}")

    return data


def step_analyze(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("═══ STEP: Correlation & Cointegration Analysis ═══")

    ub = UniverseBuilder()
    ub.load()
    liq_tickers = [t for t in ub.all_tickers() if t in data["prices"].columns]
    pairs = ub.generate_pairs(liq_tickers)

    # Correlation screen
    ca = CorrelationAnalyzer(
        min_corr=settings.analysis.correlation_min,
        window=settings.analysis.correlation_window,
    )
    high_corr = ca.high_correlation_pairs(data["returns"])
    high_corr_pairs = list(zip(high_corr["ticker_a"], high_corr["ticker_b"]))
    logger.info(f"  {len(high_corr_pairs)} pairs passed correlation filter")

    # Cointegration
    coint = CointegrationAnalyzer(
        eg_pvalue_max=settings.analysis.coint_pvalue_max,
        adf_pvalue_max=settings.analysis.adf_pvalue_max,
        half_life_min=settings.analysis.half_life_min,
        half_life_max=settings.analysis.half_life_max,
    )
    coint_df = coint.test_all_pairs(data["log_prices"], pairs)

    # Save
    coint_df.to_parquet(settings.data.processed_dir / "coint_results.parquet")
    logger.success(
        f"  {coint_df['is_cointegrated'].sum()} cointegrated pairs found"
    )
    return high_corr, coint_df


def step_train(data: dict, coint_df: pd.DataFrame) -> tuple[MLPairSelector, pd.DataFrame]:
    logger.info("═══ STEP: ML Pair Selection ═══")

    fb = FeatureBuilder(data["log_prices"], data["returns"], coint_df)
    feature_df = fb.build()
    X, y = fb.feature_matrix(feature_df)

    if len(X) < 10:
        logger.warning("Insufficient samples for ML training.")
        ranker = PairRanker(top_n=settings.ml.top_n_pairs)
        ranked = ranker.rank(coint_df)
        return None, ranked

    sel = MLPairSelector(
        model_type="xgboost",
        use_optuna=True,
        n_trials=settings.ml.optuna_trials,
        random_state=settings.ml.random_state,
    )
    sel.fit(X, y)
    cv = sel.cross_validate(X, y)
    logger.info(f"  CV AUC-ROC: {cv['auc_roc_mean']:.3f} ± {cv['auc_roc_std']:.3f}")

    ml_scores = sel.score_pairs(feature_df)
    sel.save(settings.output.reports_dir / "ml_model.joblib")

    ranker = PairRanker(top_n=settings.ml.top_n_pairs)
    ranked = ranker.rank(coint_df, ml_scores)
    ranked.to_parquet(settings.data.processed_dir / "ranked_pairs.parquet")

    return sel, ranked


def step_backtest(
    prices: pd.DataFrame, ranked: pd.DataFrame
) -> tuple[pd.DataFrame, list]:
    logger.info("═══ STEP: Backtesting ═══")

    engine = BacktestEngine(signal_method="kalman", allocation_method="equal")
    portfolio_df, pair_results = engine.run(prices, ranked)

    pv  = portfolio_df["portfolio_value"]
    ret = portfolio_df["net_return"]
    pm  = PerformanceMetrics(ret, pv, settings.backtest.risk_free_rate)
    metrics = pm.full_report()

    logger.success(
        f"  CAGR {metrics['cagr']:.1%} | Sharpe {metrics['sharpe']:.2f} | "
        f"MaxDD {metrics['max_drawdown']:.1%} | Sortino {metrics['sortino']:.2f}"
    )

    portfolio_df.to_parquet(settings.data.processed_dir / "portfolio.parquet")

    # Benchmark comparison
    bench = BenchmarkRunner(prices, settings.backtest.initial_capital)
    bench_results = bench.run_all()
    table = bench.comparison_table(metrics, bench_results)
    table_path = settings.output.reports_dir / "strategy_comparison.csv"
    table.to_csv(table_path)
    logger.info(f"  Comparison table saved → {table_path}")

    return portfolio_df, pair_results


def step_simulate(portfolio_df: pd.DataFrame) -> dict:
    logger.info("═══ STEP: Monte Carlo Simulation ═══")

    mc = MonteCarloSimulator(
        n_paths=settings.simulation.n_paths,
        horizon_days=settings.simulation.horizon_days,
        confidence_levels=settings.simulation.confidence_levels,
        random_seed=settings.simulation.random_seed,
    )

    returns = portfolio_df["net_return"]

    gbm_res  = mc.run(returns, method="gbm")
    boot_res = mc.run(returns, method="bootstrap")

    sc = ScenarioTester(returns, settings.backtest.initial_capital)
    sc_results = sc.run_all()
    sc_table = ScenarioTester.comparison_table(sc_results)
    sc_path = settings.output.reports_dir / "scenario_results.csv"
    sc_table.to_csv(sc_path)
    logger.info(f"  Scenario table saved → {sc_path}")

    logger.success(
        f"  GBM P(profit)={gbm_res['terminal_stats']['prob_profit']:.1%} | "
        f"  Bootstrap P(profit)={boot_res['terminal_stats']['prob_profit']:.1%}"
    )

    return {"gbm": gbm_res, "bootstrap": boot_res, "scenarios": sc_results}


def run_full_pipeline() -> None:
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   Statistical Arbitrage Pipeline     ║")
    logger.info("╚══════════════════════════════════════╝")

    prices  = step_collect()
    data    = step_preprocess(prices)
    _, coint_df = step_analyze(data)
    _, ranked   = step_train(data, coint_df)
    portfolio_df, _ = step_backtest(prices, ranked)
    step_simulate(portfolio_df)

    logger.success("Pipeline complete. Launch dashboard: streamlit run dashboard/app.py")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "analyze", "train", "backtest", "simulate", "full", "dashboard"],
        default="full",
        help="Pipeline mode to run (default: full).",
    )
    args = parser.parse_args()

    if args.mode == "dashboard":
        import subprocess
        subprocess.run(
            ["streamlit", "run", str(ROOT / "dashboard" / "app.py")],
            check=True,
        )
        return

    if args.mode == "collect":
        step_collect()

    elif args.mode == "analyze":
        prices = DataCollector().load_raw("prices")
        data   = step_preprocess(prices)
        step_analyze(data)

    elif args.mode == "train":
        prices = DataCollector().load_raw("prices")
        pp = Preprocessor()
        data = pp.run(prices)
        coint_df = pd.read_parquet(settings.data.processed_dir / "coint_results.parquet")
        step_train(data, coint_df)

    elif args.mode == "backtest":
        prices  = DataCollector().load_raw("prices")
        ranked  = pd.read_parquet(settings.data.processed_dir / "ranked_pairs.parquet")
        step_backtest(prices, ranked)

    elif args.mode == "simulate":
        portfolio_df = pd.read_parquet(settings.data.processed_dir / "portfolio.parquet")
        step_simulate(portfolio_df)

    else:  # full
        run_full_pipeline()


if __name__ == "__main__":
    main()
