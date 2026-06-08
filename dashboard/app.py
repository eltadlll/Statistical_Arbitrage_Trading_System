"""
dashboard/app.py
----------------
Streamlit dashboard for the Statistical Arbitrage pipeline.

Tabs:
  1. Universe & Data
  2. Correlation Analysis
  3. Cointegration & Pair Selection
  4. Strategy & Backtest
  5. Monte Carlo Simulation
  6. Strategy Comparison

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.settings import settings
from src.data.collectors import DataCollector
from src.data.preprocessors import Preprocessor
from src.data.universe import UniverseBuilder
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread import SpreadBuilder
from src.features.builder import FeatureBuilder
from src.models.ml_selector import MLPairSelector
from src.models.pair_ranker import PairRanker
from src.strategy.signals import SignalGenerator
from src.strategy.execution import ExecutionSimulator
from src.strategy.risk import RiskManager
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics
from src.backtest.benchmarks import BenchmarkRunner
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.scenario import ScenarioTester

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatArb Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1117; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px; text-align: center;
  }
  .metric-card h3 { color: #58a6ff; font-size: 1.8rem; margin: 0; }
  .metric-card p  { color: #8b949e; margin: 0; font-size: 0.85rem; }
  div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – global controls
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️  StatArb Controls")
start_date = st.sidebar.date_input("Start date", pd.to_datetime(settings.data.start_date))
end_date   = st.sidebar.date_input("End date",   pd.to_datetime(settings.data.end_date))
min_corr   = st.sidebar.slider("Min correlation",  0.50, 0.99, float(settings.analysis.correlation_min), 0.01)
coint_pval = st.sidebar.slider("Coint p-value max", 0.01, 0.10, float(settings.analysis.coint_pvalue_max), 0.01)
top_n      = st.sidebar.slider("Top-N pairs",  5, 50, settings.ml.top_n_pairs)
entry_z    = st.sidebar.slider("Entry z-score",  1.0, 3.0, float(settings.strategy.zscore_entry), 0.1)
exit_z     = st.sidebar.slider("Exit z-score",   0.0, 1.5, float(settings.strategy.zscore_exit), 0.1)
spread_method = st.sidebar.selectbox("Spread method", ["kalman", "rolling_ols", "ols"])
mc_paths   = st.sidebar.slider("MC paths",  1_000, 20_000, settings.simulation.n_paths, 1_000)
run_button = st.sidebar.button("▶  Run Full Pipeline", type="primary", use_container_width=True)

st.title("📈  Statistical Arbitrage Dashboard")
st.caption("Correlation · Cointegration · ML Selection · Backtest · Monte Carlo")

# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (cached per param combo)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading price data …")
def load_data(start: str, end: str):
    ub = UniverseBuilder()
    groups = ub.load()
    tickers = ub.all_tickers()
    collector = DataCollector(start=start, end=end)
    prices = collector.fetch(tickers)
    return prices, groups, tickers


@st.cache_data(show_spinner="Preprocessing data …")
def preprocess(prices: pd.DataFrame):
    pp = Preprocessor()
    return pp.run(prices)


@st.cache_data(show_spinner="Running correlation analysis …")
def run_correlation(returns: pd.DataFrame, _min_corr: float):
    ca = CorrelationAnalyzer(min_corr=_min_corr)
    matrix = ca.full_matrix(returns)
    high_pairs = ca.high_correlation_pairs(returns)
    return matrix, high_pairs


@st.cache_data(show_spinner="Running cointegration tests (this may take a minute) …")
def run_cointegration(log_prices: pd.DataFrame, pairs: list, pval: float):
    coint = CointegrationAnalyzer(eg_pvalue_max=pval, adf_pvalue_max=pval)
    return coint.test_all_pairs(log_prices, pairs)


@st.cache_data(show_spinner="Building ML features …")
def build_features(log_prices, returns, coint_df):
    fb = FeatureBuilder(log_prices, returns, coint_df)
    return fb.build()


@st.cache_data(show_spinner="Training ML selector …")
def train_ml(feature_df: pd.DataFrame):
    fb = FeatureBuilder.__new__(FeatureBuilder)
    X, y = FeatureBuilder.feature_matrix(fb, feature_df)
    if len(X) < 10:
        return None, None
    sel = MLPairSelector(model_type="xgboost", use_optuna=False)
    sel.fit(X, y)
    scores = sel.score_pairs(feature_df)
    return sel, scores


@st.cache_data(show_spinner="Running backtest …")
def run_backtest(prices, ranked_pairs, _entry_z, _exit_z, _method):
    engine = BacktestEngine(signal_method=_method)
    portfolio_df, pair_results = engine.run(prices, ranked_pairs)
    return portfolio_df, pair_results


@st.cache_data(show_spinner="Running Monte Carlo simulation …")
def run_mc(returns: pd.Series, _n_paths: int):
    mc = MonteCarloSimulator(n_paths=_n_paths, horizon_days=252)
    gbm_res = mc.run(returns, method="gbm")
    boot_res = mc.run(returns, method="bootstrap")
    return mc, gbm_res, boot_res


# ─────────────────────────────────────────────────────────────────────────────
# Trigger pipeline
# ─────────────────────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Loading universe …"):
        prices, groups, tickers = load_data(str(start_date), str(end_date))
        st.session_state["prices"] = prices
        st.session_state["groups"] = groups

    with st.spinner("Preprocessing …"):
        data = preprocess(prices)
        st.session_state["data"] = data

    with st.spinner("Correlation …"):
        corr_matrix, high_pairs = run_correlation(data["returns"], min_corr)
        st.session_state["corr_matrix"] = corr_matrix
        st.session_state["high_pairs"] = high_pairs

    # Generate candidate pairs
    ub = UniverseBuilder()
    ub.load()
    liq_tickers = [t for t in ub.all_tickers() if t in prices.columns]
    candidate_pairs = ub.generate_pairs(liq_tickers)

    with st.spinner("Cointegration …"):
        coint_df = run_cointegration(data["log_prices"], candidate_pairs, coint_pval)
        st.session_state["coint_df"] = coint_df

    with st.spinner("ML …"):
        feature_df = build_features(data["log_prices"], data["returns"], coint_df)
        ml_sel, ml_scores = train_ml(feature_df)
        st.session_state["feature_df"] = feature_df
        st.session_state["ml_scores"] = ml_scores

    with st.spinner("Ranking pairs …"):
        ranker = PairRanker(top_n=top_n)
        ranked = ranker.rank(coint_df, ml_scores)
        st.session_state["ranked_pairs"] = ranked

    with st.spinner("Backtesting …"):
        portfolio_df, pair_results = run_backtest(
            prices, ranked, entry_z, exit_z, spread_method
        )
        st.session_state["portfolio_df"] = portfolio_df
        st.session_state["pair_results"] = pair_results

    with st.spinner("Monte Carlo …"):
        mc, gbm_res, boot_res = run_mc(portfolio_df["net_return"], mc_paths)
        st.session_state["mc"] = mc
        st.session_state["gbm_res"] = gbm_res
        st.session_state["boot_res"] = boot_res

    with st.spinner("Benchmarks …"):
        bench = BenchmarkRunner(prices, settings.backtest.initial_capital)
        bench_results = bench.run_all()
        st.session_state["bench_results"] = bench_results

    st.success("Pipeline complete!", icon="✅")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📦 Universe & Data",
    "🔗 Correlation",
    "⚗️ Cointegration & ML",
    "📊 Backtest",
    "🎲 Monte Carlo",
    "⚖️ Comparison",
])

# ──────────────────────────── TAB 1: Universe & Data ────────────────────────
with tab1:
    st.header("Universe & Raw Data")
    prices = st.session_state.get("prices")
    groups = st.session_state.get("groups", {})

    if prices is None:
        st.info("Click **▶ Run Full Pipeline** in the sidebar to load data.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers loaded", prices.shape[1])
        c2.metric("Trading days", prices.shape[0])
        c3.metric("Start", str(prices.index[0].date()))
        c4.metric("End", str(prices.index[-1].date()))

        st.subheader("Sector breakdown")
        sector_counts = {k: len(v) for k, v in groups.items()}
        fig = px.bar(
            x=list(sector_counts.keys()),
            y=list(sector_counts.values()),
            labels={"x": "Sector", "y": "# Tickers"},
            color=list(sector_counts.values()),
            color_continuous_scale="Blues",
            title="Tickers per Sector",
        )
        fig.update_layout(showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalised price chart")
        sel_tickers = st.multiselect(
            "Select tickers to plot",
            options=prices.columns.tolist(),
            default=prices.columns[:8].tolist(),
        )
        if sel_tickers:
            rebased = (prices[sel_tickers] / prices[sel_tickers].iloc[0]) * 100
            fig2 = px.line(rebased, title="Rebased Prices (base = 100)",
                           template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Summary statistics")
        data = st.session_state.get("data", {})
        if "returns" in data:
            desc = data["returns"].describe().T[["mean", "std", "min", "max"]]
            desc.columns = ["Daily Mean", "Daily Std", "Min", "Max"]
            st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)


# ──────────────────────────── TAB 2: Correlation ────────────────────────────
with tab2:
    st.header("Correlation Analysis")
    corr_matrix = st.session_state.get("corr_matrix")
    high_pairs  = st.session_state.get("high_pairs")

    if corr_matrix is None:
        st.info("Run the pipeline first.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Full correlation heatmap")
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                title="Pairwise Pearson Correlation",
                template="plotly_dark",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader(f"Pairs |r| ≥ {min_corr}")
            if not high_pairs.empty:
                st.dataframe(
                    high_pairs.head(30).style.background_gradient(
                        subset=["correlation"], cmap="Blues"
                    ),
                    use_container_width=True,
                )

        st.subheader("Rolling correlation explorer")
        data = st.session_state.get("data", {})
        if "returns" in data:
            ret = data["returns"]
            avail = ret.columns.tolist()
            col1, col2 = st.columns(2)
            ta = col1.selectbox("Ticker A", avail, index=0)
            tb = col2.selectbox("Ticker B", avail, index=min(1, len(avail)-1))
            roll_win = st.slider("Rolling window (days)", 20, 252, 60)
            if ta != tb:
                roll = ret[ta].rolling(roll_win).corr(ret[tb])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=roll.index, y=roll.values,
                                          mode="lines", name="Rolling Corr"))
                fig3.add_hline(y=min_corr, line_dash="dot", line_color="green",
                               annotation_text=f"threshold={min_corr}")
                fig3.add_hline(y=-min_corr, line_dash="dot", line_color="red")
                fig3.update_layout(
                    title=f"Rolling {roll_win}d Correlation: {ta} vs {tb}",
                    yaxis_title="Pearson r", template="plotly_dark"
                )
                st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────── TAB 3: Cointegration & ML ─────────────────────
with tab3:
    st.header("Cointegration & ML Pair Selection")
    coint_df = st.session_state.get("coint_df")
    ranked   = st.session_state.get("ranked_pairs")
    ml_scores = st.session_state.get("ml_scores")

    if coint_df is None:
        st.info("Run the pipeline first.")
    else:
        # Summary metrics
        n_pass = int(coint_df["is_cointegrated"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pairs tested", len(coint_df))
        c2.metric("Cointegrated", n_pass)
        c3.metric("Pass rate", f"{n_pass/max(len(coint_df),1):.1%}")
        c4.metric("Top pairs selected", len(ranked) if ranked is not None else 0)

        st.subheader("Cointegration results")
        display_cols = ["ticker_a", "ticker_b", "eg_pvalue", "adf_pvalue",
                        "half_life", "hedge_ratio", "is_cointegrated"]
        st.dataframe(
            coint_df[[c for c in display_cols if c in coint_df.columns]]
            .head(50)
            .style.applymap(
                lambda v: "background-color: #1a3a1a" if v is True else
                          "background-color: #3a1a1a" if v is False else "",
                subset=["is_cointegrated"],
            ),
            use_container_width=True,
        )

        # Scatter: half-life vs p-value
        st.subheader("Half-life vs EG p-value")
        plot_df = coint_df[coint_df["half_life"].between(0, 120)].copy()
        plot_df["cointegrated"] = plot_df["is_cointegrated"].map(
            {True: "Yes", False: "No"}
        )
        fig = px.scatter(
            plot_df, x="half_life", y="eg_pvalue",
            color="cointegrated",
            color_discrete_map={"Yes": "#3fb950", "No": "#f85149"},
            hover_data=["ticker_a", "ticker_b"],
            title="Half-life vs EG p-value",
            template="plotly_dark",
        )
        fig.add_hline(y=coint_pval, line_dash="dot", annotation_text=f"p={coint_pval}")
        st.plotly_chart(fig, use_container_width=True)

        # Spread explorer
        if ranked is not None and not ranked.empty:
            st.subheader("Spread & Z-score explorer")
            pair_options = [
                f"{r['ticker_a']}|{r['ticker_b']}" for _, r in ranked.iterrows()
            ]
            chosen = st.selectbox("Select pair", pair_options)
            ta, tb = chosen.split("|")
            prices = st.session_state.get("prices")
            if prices is not None and ta in prices.columns and tb in prices.columns:
                sb = SpreadBuilder()
                lp_a = np.log(prices[ta])
                lp_b = np.log(prices[tb])
                spread, beta_k, _ = sb.kalman_spread(lp_a, lp_b)
                z = sb.zscore(spread)

                fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=["Spread", "Z-score"])
                fig4.add_trace(go.Scatter(x=spread.index, y=spread,
                                          mode="lines", name="Spread"), row=1, col=1)
                fig4.add_trace(go.Scatter(x=z.index, y=z,
                                          mode="lines", name="Z-score"), row=2, col=1)
                fig4.add_hline(y=entry_z, row=2, col=1, line_color="green",
                               line_dash="dot")
                fig4.add_hline(y=-entry_z, row=2, col=1, line_color="red",
                               line_dash="dot")
                fig4.add_hline(y=0, row=2, col=1, line_color="gray", line_dash="dash")
                fig4.update_layout(template="plotly_dark", height=500,
                                   title=f"Kalman Spread: {ta} vs {tb}")
                st.plotly_chart(fig4, use_container_width=True)

        # ML scores
        if ml_scores is not None and not ml_scores.empty:
            st.subheader("ML pair scores (top 20)")
            fig5 = px.bar(
                ml_scores.head(20),
                x=ml_scores.head(20).apply(
                    lambda r: f"{r['ticker_a']}|{r['ticker_b']}", axis=1
                ),
                y="ml_score",
                color="ml_score",
                color_continuous_scale="Greens",
                title="ML Score by Pair",
                template="plotly_dark",
            )
            st.plotly_chart(fig5, use_container_width=True)


# ──────────────────────────── TAB 4: Backtest ───────────────────────────────
with tab4:
    st.header("Strategy Backtest")
    portfolio_df = st.session_state.get("portfolio_df")
    pair_results = st.session_state.get("pair_results", [])

    if portfolio_df is None:
        st.info("Run the pipeline first.")
    else:
        pv = portfolio_df["portfolio_value"]
        ret = portfolio_df["net_return"]
        pm = PerformanceMetrics(ret, pv, settings.backtest.risk_free_rate)
        metrics = pm.full_report()

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return",  f"{metrics['total_return']:.1%}")
        c2.metric("CAGR",          f"{metrics['cagr']:.1%}")
        c3.metric("Sharpe",        f"{metrics['sharpe']:.2f}")
        c4.metric("Max Drawdown",  f"{metrics['max_drawdown']:.1%}")
        c5.metric("Sortino",       f"{metrics['sortino']:.2f}")

        # Equity curve
        st.subheader("Portfolio equity curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values,
            mode="lines", name="StatArb Portfolio",
            line=dict(color="#58a6ff", width=2),
        ))
        fig.add_hline(y=settings.backtest.initial_capital,
                      line_dash="dot", line_color="gray")
        fig.update_layout(
            template="plotly_dark", height=400,
            title="Portfolio Equity Curve",
            yaxis_title="Portfolio Value ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        st.subheader("Drawdown")
        dd_series = pm.drawdown_series()
        fig_dd = px.area(
            x=dd_series.index, y=dd_series.values * 100,
            labels={"x": "Date", "y": "Drawdown (%)"},
            template="plotly_dark", title="Running Drawdown",
            color_discrete_sequence=["#f85149"],
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Rolling Sharpe
        st.subheader("Rolling 63-day Sharpe")
        roll_sharpe = pm.rolling_sharpe(63)
        fig_rs = px.line(x=roll_sharpe.index, y=roll_sharpe.values,
                         template="plotly_dark", title="Rolling Sharpe (63d)",
                         labels={"x": "Date", "y": "Sharpe"})
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_rs, use_container_width=True)

        # Per-pair metrics
        if pair_results:
            st.subheader("Per-pair performance")
            pair_rows = [
                {"Pair": r.pair, **r.metrics}
                for r in pair_results
                if r.metrics
            ]
            pair_table = pd.DataFrame(pair_rows).set_index("Pair")
            st.dataframe(
                pair_table.style.format({
                    c: "{:.2f}" if "sharpe" in c.lower() or "sortino" in c.lower()
                    else "{:.1%}" for c in pair_table.columns
                }),
                use_container_width=True,
            )


# ──────────────────────────── TAB 5: Monte Carlo ────────────────────────────
with tab5:
    st.header("Monte Carlo Simulation")
    gbm_res  = st.session_state.get("gbm_res")
    boot_res = st.session_state.get("boot_res")
    mc       = st.session_state.get("mc")

    if gbm_res is None:
        st.info("Run the pipeline first.")
    else:
        for res, label in [(gbm_res, "GBM"), (boot_res, "Block Bootstrap")]:
            st.subheader(f"{label} Simulation – {mc_paths:,} paths")
            bands = res["bands"]
            ts = res["terminal_stats"]
            ds = res["drawdown_stats"]

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("E[terminal value]", f"${ts['mean_terminal_value']:,.0f}")
            c2.metric("P(profit)",          f"{ts['prob_profit']:.1%}")
            c3.metric("VaR 95%",            f"{ts.get('var_95', 0):.1%}")
            c4.metric("CVaR 95%",           f"{ts.get('cvar_95', 0):.1%}")

            # Fan chart
            fig = go.Figure()
            x = list(range(len(bands)))
            fig.add_trace(go.Scatter(x=x + x[::-1],
                                     y=list(bands["p5"]) + list(bands["p95"])[::-1],
                                     fill="toself", fillcolor="rgba(88,166,255,0.1)",
                                     line=dict(color="rgba(255,255,255,0)"),
                                     name="5th–95th pct"))
            fig.add_trace(go.Scatter(x=x + x[::-1],
                                     y=list(bands["p25"]) + list(bands["p75"])[::-1],
                                     fill="toself", fillcolor="rgba(88,166,255,0.2)",
                                     line=dict(color="rgba(255,255,255,0)"),
                                     name="25th–75th pct"))
            fig.add_trace(go.Scatter(x=x, y=bands["p50"], mode="lines",
                                     name="Median", line=dict(color="#58a6ff", width=2)))
            fig.add_hline(y=settings.backtest.initial_capital,
                          line_dash="dot", line_color="gray")
            fig.update_layout(
                template="plotly_dark", height=400,
                title=f"{label}: Portfolio Value Fan Chart",
                xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Terminal distribution
            terminal = res["paths"][:, -1]
            fig2 = px.histogram(
                terminal, nbins=100,
                title=f"{label}: Terminal Value Distribution",
                labels={"value": "Terminal Value ($)"},
                template="plotly_dark",
                color_discrete_sequence=["#58a6ff"],
            )
            fig2.add_vline(x=settings.backtest.initial_capital,
                           line_dash="dot", annotation_text="Initial capital")
            st.plotly_chart(fig2, use_container_width=True)

        # Scenario stress tests
        st.subheader("Stress-test scenarios")
        portfolio_df = st.session_state.get("portfolio_df")
        if portfolio_df is not None:
            st_tester = ScenarioTester(portfolio_df["net_return"],
                                       settings.backtest.initial_capital)
            sc_results = st_tester.run_all()
            sc_table = ScenarioTester.comparison_table(sc_results)
            st.dataframe(sc_table, use_container_width=True)

            fig3 = go.Figure()
            for sc in sc_results:
                fig3.add_trace(go.Scatter(
                    x=sc.portfolio_value.index,
                    y=sc.portfolio_value.values,
                    mode="lines", name=sc.name,
                ))
            fig3.update_layout(
                template="plotly_dark", height=450,
                title="Scenario Equity Curves",
                yaxis_title="Portfolio Value ($)",
            )
            st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────── TAB 6: Comparison ─────────────────────────────
with tab6:
    st.header("Strategy Comparison")
    bench_results = st.session_state.get("bench_results")
    portfolio_df  = st.session_state.get("portfolio_df")

    if bench_results is None or portfolio_df is None:
        st.info("Run the pipeline first.")
    else:
        pv  = portfolio_df["portfolio_value"]
        ret = portfolio_df["net_return"]
        pm  = PerformanceMetrics(ret, pv, settings.backtest.risk_free_rate)
        statarb_metrics = pm.full_report()

        bench = BenchmarkRunner(
            st.session_state.get("prices", pd.DataFrame()),
            settings.backtest.initial_capital,
        )
        table = bench.comparison_table(statarb_metrics, bench_results)
        st.subheader("Metrics comparison table")
        st.dataframe(
            table.style.highlight_max(subset=["sharpe", "cagr", "total_return"],
                                      color="#1a3a1a")
                       .highlight_min(subset=["max_drawdown"], color="#1a3a1a")
                       .format({c: "{:.2%}" if c not in ["sharpe","sortino","calmar"]
                                else "{:.2f}" for c in table.columns}),
            use_container_width=True,
        )

        # Equity curve overlay
        st.subheader("Equity curves – all strategies")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv.values,
            name="StatArb", line=dict(color="#58a6ff", width=3),
        ))
        for bname, bdata in bench_results.items():
            curve = bdata.get("curve")
            if curve is not None and not curve.empty:
                fig.add_trace(go.Scatter(
                    x=curve.index, y=curve.values,
                    name=bdata["name"],
                    line=dict(dash="dot"),
                ))
        fig.add_hline(y=settings.backtest.initial_capital,
                      line_dash="dot", line_color="gray")
        fig.update_layout(
            template="plotly_dark", height=500,
            title="Equity Curves Comparison",
            yaxis_title="Portfolio Value ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart of risk metrics
        st.subheader("Risk-return radar")
        radar_metrics = ["sharpe", "sortino", "calmar"]
        radar_data = {"StatArb": [statarb_metrics.get(m, 0) for m in radar_metrics]}
        for bname, bdata in bench_results.items():
            radar_data[bdata["name"]] = [
                bdata["metrics"].get(m, 0) for m in radar_metrics
            ]

        fig_r = go.Figure()
        for strategy, vals in radar_data.items():
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself", name=strategy,
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark",
            title="Risk-Adjusted Metrics Radar",
        )
        st.plotly_chart(fig_r, use_container_width=True)
