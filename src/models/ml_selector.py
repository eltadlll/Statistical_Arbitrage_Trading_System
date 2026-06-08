"""
models/ml_selector.py
---------------------
Trains an XGBoost or Random Forest classifier to score candidate pairs.
Uses Optuna for hyperparameter tuning and SHAP for feature importance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import optuna
from joblib import dump, load

from config.settings import settings

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLPairSelector:
    """
    Binary classifier that predicts whether a pair will be
    profitable for statistical arbitrage.

    Parameters
    ----------
    model_type : "xgboost" | "random_forest"
    use_optuna : Whether to run hyperparameter search before fitting.
    n_trials   : Number of Optuna trials if use_optuna=True.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        use_optuna: bool = True,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []
        self.shap_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPairSelector":
        """Train the model. Runs Optuna search if use_optuna=True."""
        self.feature_names = X.columns.tolist()
        X_arr = X.values.astype(np.float32)
        y_arr = y.values

        if self.use_optuna:
            logger.info(f"Running Optuna with {self.n_trials} trials …")
            best_params = self._optuna_search(X_arr, y_arr)
        else:
            best_params = {}

        clf = self._build_model(best_params)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        self.pipeline.fit(X_arr, y_arr)
        logger.info("Model trained.")
        return self

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, float]:
        """Return cross-validated AUC-ROC and average precision."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() before cross_validate().")
        X_arr = X.values.astype(np.float32)
        y_arr = y.values

        auc_scores = cross_val_score(
            self.pipeline, X_arr, y_arr,
            cv=StratifiedKFold(cv, shuffle=True, random_state=self.random_state),
            scoring="roc_auc",
        )
        ap_scores = cross_val_score(
            self.pipeline, X_arr, y_arr,
            cv=StratifiedKFold(cv, shuffle=True, random_state=self.random_state),
            scoring="average_precision",
        )
        result = {
            "auc_roc_mean": float(auc_scores.mean()),
            "auc_roc_std": float(auc_scores.std()),
            "avg_precision_mean": float(ap_scores.mean()),
            "avg_precision_std": float(ap_scores.std()),
        }
        logger.info(
            f"CV results | AUC-ROC {result['auc_roc_mean']:.3f} ± {result['auc_roc_std']:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of class 1 (good pair) for each row."""
        if self.pipeline is None:
            raise RuntimeError("Model not fitted.")
        return self.pipeline.predict_proba(X.values.astype(np.float32))[:, 1]

    def score_pairs(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'ml_score' column to a feature DataFrame.
        Higher score = more likely to be a good pair.
        """
        available = [c for c in self.feature_names if c in feature_df.columns]
        X = feature_df[available].replace([np.inf, -np.inf], np.nan).fillna(0)
        probs = self.predict_proba(X)
        out = feature_df[["ticker_a", "ticker_b"]].copy()
        out["ml_score"] = probs
        return out.sort_values("ml_score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------

    def compute_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values and return feature importance summary."""
        if self.pipeline is None:
            raise RuntimeError("Model not fitted.")
        clf = self.pipeline.named_steps["clf"]
        scaler = self.pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X.values.astype(np.float32))

        explainer = shap.TreeExplainer(clf)
        self.shap_values = explainer.shap_values(X_scaled)

        vals = self.shap_values
        if isinstance(vals, list):
            vals = vals[1]  # class 1 for binary

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        logger.info("Top SHAP features:\n" + importance.head(10).to_string(index=False))
        return importance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        dump(self.pipeline, path)
        logger.info(f"Model saved → {path}")

    def load(self, path: Path) -> "MLPairSelector":
        self.pipeline = load(path)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, params: dict):
        if self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 400),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                min_child_weight=params.get("min_child_weight", 1),
                gamma=params.get("gamma", 0.0),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
                verbosity=0,
            )
        else:  # random_forest
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 400),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                max_features=params.get("max_features", "sqrt"),
                random_state=self.random_state,
                n_jobs=-1,
            )

    def _optuna_search(self, X: np.ndarray, y: np.ndarray) -> dict:
        def objective(trial):
            if self.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                }

            clf = self._build_model(params)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            scores = cross_val_score(
                pipe, X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring="roc_auc",
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"Best Optuna AUC-ROC: {study.best_value:.4f} | params: {study.best_params}")
        return study.best_params
