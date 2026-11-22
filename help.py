"""
Evaluation harness for Swissgrid control-power marginal price forecasting.

Usage:
    from src.evaluation import Metrics, WalkForwardSplitter, Evaluator, PredictionValidator

    # Métriques individuelles
    Metrics.mae(y_true, y_pred)
    Metrics.rmse(y_true, y_pred)
    Metrics.pinball(y_true, y_pred, q=0.5)
    Metrics.coverage(y_true, lo, hi)

    # Découpage walk-forward
    splitter = WalkForwardSplitter(start_test="2025-01-01", freq="M")
    for train, test in splitter.split(df):
        ...

    # Évaluation complète d'un fichier de prédictions
    evaluator = Evaluator(actuals=df_actuals)
    report = evaluator.evaluate(preds=df_preds)
    print(report)

    # Validation du format d'un fichier de prédictions
    PredictionValidator.validate("predictions_lgbm.parquet")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------

class Metrics:
    """Ensemble des métriques du contrat d'évaluation. Toutes les entrées
    sont supposées être en log10 du prix marginal (CHF/MW)."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error en log10."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error en log10."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    @staticmethod
    def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        """Pinball loss à quantile q (0 < q < 1)."""
        if not 0 < q < 1:
            raise ValueError(f"q doit être dans (0, 1), reçu {q}")
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        d = y_true - y_pred
        return float(np.mean(np.maximum(q * d, (q - 1) * d)))

    @staticmethod
    def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
        """Part des réels dans [lo, hi] (cible ≈ 0.80 pour p10–p90)."""
        y_true, lo, hi = np.asarray(y_true), np.asarray(lo), np.asarray(hi)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))


# ---------------------------------------------------------------------------
# 2. Walk-forward splitter
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplitter:
    """Découpage walk-forward du roadmap : test fixe à `start_test`,
    retrain au début de chaque période de longueur `freq` (par défaut mensuel).

    Hypothèses :
    - df contient une colonne datetime UTC `avail` (gate_closure + 2h)
      et une colonne datetime UTC `gc` (gate_closure_time_utc).
    - L'ordre chronologique est garanti par tri sur `avail` / `gc`.
    """

    start_test: str = "2025-01-01"
    end_test: str | None = None  # None = jusqu'à la dernière ligne
    freq: str = "M"              # "M" mensuel, "W" hebdo, "Q" trimestriel
    avail_col: str = "avail"
    gc_col: str = "gc"

    def _to_utc(self, s: str) -> pd.Timestamp:
        return pd.Timestamp(s, tz="UTC") if pd.Timestamp(s).tz is None else pd.Timestamp(s)

    def splits(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Génère (train, test) pour chaque période de test."""
        t0 = self._to_utc(self.start_test)
        t_end = self._to_utc(self.end_test) if self.end_test else df[self.gc_col].max()

        # bornes des périodes de test
        periods = pd.period_range(t0, t_end, freq=self.freq)
        for period in periods:
            p_start = period.start_time.tz_localize("UTC")
            p_end = period.end_time.tz_localize("UTC")

            train = df[df[self.avail_col] < p_start]
            test = df[(df[self.gc_col] >= p_start) & (df[self.gc_col] <= p_end)]

            if test.empty:
                continue
            yield train, test


# ---------------------------------------------------------------------------
# 3. Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvaluationRow:
    """Une ligne du rapport d'évaluation, par (model, frequency, direction)."""
    model: str
    frequency: str
    direction: str
    n: int
    mae: float
    rmse: float
    pinball_q10: float | None = None
    pinball_q50: float | None = None
    pinball_q90: float | None = None
    coverage_80: float | None = None


@dataclass
class Evaluator:
    """Évalue un ou plusieurs fichiers de prédictions contre les résultats réels.

    `actuals` doit être le DataFrame d'enchères avec :
        - auction_id
        - y (log10 du prix marginal, NaN pour les enchères sans achat)
        - frequency  (daily / weekly)
        - direction  (upwards / downwards)
        - gc         (gate_closure_time_utc, UTC)

    Convention : les lignes avec `y` NaN (enchères sans achat) ne sont **pas**
    scorées ; elles doivent toutefois être présentes dans le fichier de
    prédictions (vérifié par PredictionValidator).
    """

    actuals: pd.DataFrame
    group_cols: tuple[str, ...] = ("frequency", "direction")

    def _merge(self, preds: pd.DataFrame) -> pd.DataFrame:
        missing = {"auction_id", "marginal_price_pred", "model"} - set(preds.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans preds : {missing}")

        # predictions en log10
        p = preds.copy()
        p["log10_pred"] = np.log10(p["marginal_price_pred"].astype(float))
        if {"marginal_price_p10", "marginal_price_p90"}.issubset(p.columns):
            p["log10_p10"] = np.log10(p["marginal_price_p10"].astype(float))
            p["log10_p90"] = np.log10(p["marginal_price_p90"].astype(float))

        merged = self.actuals.merge(
            p, on="auction_id", how="inner", suffixes=("", "_pred")
        )
        # on ne score que les enchères avec une target
        merged = merged[merged["y"].notna()]
        if merged.empty:
            raise ValueError(
                "Aucune ligne joignable après merge : vérifier la période "
                "de test et le nom des colonnes."
            )
        return merged

    def evaluate(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Retourne un DataFrame avec une ligne par (model, frequency, direction)."""
        m = self._merge(preds)
        rows: list[EvaluationRow] = []

        for (model, freq, direction), g in m.groupby(["model", *self.group_cols]):
            y = g["y"].to_numpy()
            y_hat = g["log10_pred"].to_numpy()
            row = EvaluationRow(
                model=model, frequency=freq, direction=direction, n=len(g),
                mae=Metrics.mae(y, y_hat),
                rmse=Metrics.rmse(y, y_hat),
            )
            if "log10_p10" in g.columns and "log10_p90" in g.columns:
                row.pinball_q10 = Metrics.pinball(y, g["log10_p10"].to_numpy(), 0.1)
                row.pinball_q50 = Metrics.pinball(y, y_hat, 0.5)
                row.pinball_q90 = Metrics.pinball(y, g["log10_p90"].to_numpy(), 0.9)
                row.coverage_80 = Metrics.coverage(
                    y, g["log10_p10"].to_numpy(), g["log10_p90"].to_numpy()
                )
            rows.append(row)

        return pd.DataFrame([r.__dict__ for r in rows])

    def evaluate_many(self, preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Évalue plusieurs modèles en un coup (clé = nom du modèle)."""
        reports = []
        for name, preds in preds_by_model.items():
            p = preds.copy()
            # on force le nom du modèle à la clé pour éviter les confusions
            p["model"] = name
            reports.append(self.evaluate(p))
        return pd.concat(reports, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Prediction validator (contrat de format)
# ---------------------------------------------------------------------------

@dataclass
class PredictionValidator:
    """Vérifie qu'un fichier de prédictions respecte le contrat."""

    required: tuple[str, ...] = ("auction_id", "marginal_price_pred", "model")
    recommended: tuple[str, ...] = ("marginal_price_p10", "marginal_price_p90")

    @classmethod
    def validate(cls, path: str) -> None:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

        for c in cls.required:
            if c not in df.columns:
                raise ValueError(f"colonne obligatoire manquante : {c}")

        if df["marginal_price_pred"].isna().any():
            raise ValueError("NaN dans marginal_price_pred")
        if (df["marginal_price_pred"] <= 0).any():
            raise ValueError("marginal_price_pred <= 0")
        if df["auction_id"].duplicated().any():
            dup = df.loc[df["auction_id"].duplicated(), "auction_id"].unique()[:5]
            raise ValueError(f"auction_id dupliqué (exemples : {dup})")

        if all(c in df.columns for c in cls.recommended):
            if (df["marginal_price_p10"] > df["marginal_price_pred"]).any():
                raise ValueError("p10 > pred pour certaines lignes")
            if (df["marginal_price_pred"] > df["marginal_price_p90"]).any():
                raise ValueError("pred > p90 pour certaines lignes")

        print(f"[OK] {path} : {len(df)} lignes, "
              f"{df['model'].nunique()} modèle(s), "
              f"{df['auction_id'].nunique()} auctions uniques")