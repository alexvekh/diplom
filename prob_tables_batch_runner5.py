# prob_tables Batch_runner.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

import prob_tables as pt


# ============================================================================
# CONFIG: edit these lists and parameters for your experiments
# ============================================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMD",
    "DIS"
]

MARKET = "SPY"

# Different history windows to test
PERIODS = [
    {"name": "2010_now", "start": "2010-01-01", "end": None},
    {"name": "2015_now", "start": "2015-01-01", "end": None},
    {"name": "2020_now", "start": "2020-01-01", "end": None},
]

# Parameter grid to test. Add/remove values as needed.
PARAM_GRID = {
    "horizon": [10, 20, 30],
    "theta_mode": ["atr", "vol"],
    "theta_k": [0.25, 0.40],
    "lookback_high": [252],
    "alpha": [1.0],
}

# Feature weights for log-odds aggregation
WEIGHTS = {
    "dist": 0.12,
    "rsi": 0.12,
    "macd": 0.08,
    "rsi_regime": 0.1,
    "vol": 0.07,
    "trend": 0.07,
    "market": 0.12,
    "mom": 0.10,
    "ma200": 0.10,
    "volpct": 0.12,
    "spy_mom": 0.10,
}

TRAIN_FRAC = 0.80
BASE_RATE_MODE = "empirical"  # "empirical" or "uniform"
SAVE_ROW_LEVEL = True
OUTPUT_DIR = Path("artifacts/prob_tables_batch")


# ============================================================================
# Helpers
# ============================================================================

FEATURE_KEYS = [
    "dist", "rsi", "rsi_regime", "macd", "vol", "trend",
    "market", "mom", "ma200", "volpct", "spy_mom",
]

FEATURE_TO_BUCKET_COL = {
    "dist": "dist_bucket",
    "rsi": "rsi_bucket",
    "rsi_regime": "rsi_regime_bucket",
    "macd": "macd_bucket",
    "vol": "vol_bucket",
    "trend": "trend_bucket",
    "market": "market_bucket",
    "mom": "mom_bucket",
    "ma200": "ma200_bucket",
    "volpct": "volpct_bucket",
    "spy_mom": "spy_mom_bucket",
}

CLASS_LABELS = [0, 1, 2]  # 0=Decline, 1=Sideways, 2=Growth
CLASS_NAMES = {0: "Decline", 1: "Sideways", 2: "Growth"}


def brier_multiclass(y_true: np.ndarray, proba_dsg: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(proba_dsg, dtype=float)
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))


def get_base_rate(y: pd.Series, mode: str = "empirical") -> np.ndarray:
    if mode == "uniform":
        base = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    elif mode == "empirical":
        base = y.value_counts(normalize=True).reindex(CLASS_LABELS).fillna(0.0).values.astype(float)
        if base.sum() <= 0:
            base = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        raise ValueError("mode must be 'empirical' or 'uniform'")

    base = np.clip(base, 1e-12, 1.0)
    base = base / base.sum()
    return base


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    return pt.normalize_weights(weights, FEATURE_KEYS)


def iter_param_grid(grid: dict[str, list[Any]]):
    import itertools

    keys = list(grid.keys())
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def build_ready_df(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    horizon: int,
    lookback_high: int,
    theta_mode: str,
    theta_k: float,
) -> pd.DataFrame:
    s = pt.create_stock_features(stock_df)
    m = pt.create_market_features(market_df)

    s = s.join(m[["spy_ret", "spy_trend_regime", "spy_dd_pct", "spy_volatility"]], how="left")

    s["future_ret"] = s["close"].shift(-horizon) / s["close"] - 1.0

    scale = np.sqrt(horizon)
    if theta_mode == "atr":
        s["theta"] = theta_k * s["atr_pct"] * scale
    else:
        s["theta"] = theta_k * s["volatility_20"] * scale

    s["target_3"] = pt.make_target_3(s["future_ret"], s["theta"])

    roll_high = s["close"].rolling(lookback_high).max()
    s["dd_pct"] = (roll_high - s["close"]) / roll_high * 100.0

    s["dist_bucket"] = pt.bucket_distance_from_high(s["dd_pct"])
    s["rsi_bucket"] = pt.bucket_rsi(s["rsi"])
    s["macd_bucket"] = pt.macd_status(s["close"])
    s["trend_bucket"] = pt.trend_bucket(s["close"], s["ma200"], s["ma50"])
    s["vol_bucket"] = pt.bucket_volume(s["vol_ratio"], s["close"], s["open"])
    s["market_bucket"] = pt.market_bucket(s["spy_trend_regime"], s["spy_dd_pct"])

    s["mom20"] = s["close"].pct_change(20)
    s["mom_bucket"] = pt.bucket_momentum(s["mom20"])

    s["dist_ma200"] = (s["close"] - s["ma200"]) / s["ma200"]
    s["ma200_bucket"] = pt.bucket_dist_ma200(s["dist_ma200"])
    s["volpct_bucket"] = pt.bucket_vol_percentile(s["volatility_20"], window=252)
    s["rsi_regime_bucket"] = (s["rsi_bucket"].astype(str) + "_" + s["spy_trend_regime"].astype(str))

    spy_mom_bucket = pt.bucket_spy_momentum(m["close"], lookback=60)
    s = s.join(spy_mom_bucket.rename("spy_mom_bucket"), how="left")

    return s.dropna().copy()



def build_tables(train_df: pd.DataFrame, alpha: float) -> dict[str, pd.DataFrame]:
    return {
        "dist": pt.prob_table_3class(train_df["dist_bucket"], train_df["target_3"], alpha=alpha),
        "rsi": pt.prob_table_3class(train_df["rsi_bucket"], train_df["target_3"], alpha=alpha),
        "macd": pt.prob_table_3class(train_df["macd_bucket"], train_df["target_3"], alpha=alpha),
        "vol": pt.prob_table_3class(train_df["vol_bucket"], train_df["target_3"], alpha=alpha),
        "trend": pt.prob_table_3class(train_df["trend_bucket"], train_df["target_3"], alpha=alpha),
        "market": pt.prob_table_3class(train_df["market_bucket"], train_df["target_3"], alpha=alpha),
        "mom": pt.prob_table_3class(train_df["mom_bucket"], train_df["target_3"], alpha=alpha),
        "ma200": pt.prob_table_3class(train_df["ma200_bucket"], train_df["target_3"], alpha=alpha),
        "volpct": pt.prob_table_3class(train_df["volpct_bucket"], train_df["target_3"], alpha=alpha),
        "spy_mom": pt.prob_table_3class(train_df["spy_mom_bucket"], train_df["target_3"], alpha=alpha),
        "rsi_regime": pt.prob_table_3class(train_df["rsi_regime_bucket"], train_df["target_3"], alpha=alpha),
    }



def predict_tables_logodds(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float],
    base_rate_dsg: np.ndarray,
    fallback_dsg: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    w = normalize_weights(weights)

    base = np.asarray(base_rate_dsg, dtype=float)
    base = np.clip(base, 1e-12, 1.0)
    base = base / base.sum()

    fallback = np.asarray(fallback_dsg, dtype=float)
    fallback = np.clip(fallback, 1e-12, 1.0)
    fallback = fallback / fallback.sum()

    out = np.zeros((len(df), 3), dtype=float)  # [D,S,G]
    scores_out = np.zeros((len(df), 3), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        score = np.zeros(3, dtype=float)

        for feature in FEATURE_KEYS:
            bucket_col = FEATURE_TO_BUCKET_COL[feature]
            bucket_value = row[bucket_col]
            tbl = tables[feature]

            if bucket_value in tbl.index:
                p = tbl.loc[bucket_value][["Decline", "Sideways", "Growth"]].values.astype(float)
            else:
                p = fallback.copy()

            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum()

            score += w[feature] * np.log(p / base)

        z = score - np.max(score)
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum()

        out[i] = probs
        scores_out[i] = score

    score_df = pd.DataFrame(scores_out, index=df.index, columns=["S_Decline", "S_Sideways", "S_Growth"])
    return out, score_df



def summarize_rows(
    pred_df: pd.DataFrame,
    *,
    ticker: str,
    period_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_rate_mode: str,
) -> dict[str, Any]:
    y_true = pred_df["true_class"].astype(int).values
    p = pred_df[["P_Decline", "P_Sideways", "P_Growth"]].values
    y_pred = pred_df["predicted_class"].astype(int).values

    row: dict[str, Any] = {
        "ticker": ticker,
        "period": period_name,
        "start_date": str(test_df.index.min().date()) if len(test_df) else None,
        "end_date": str(test_df.index.max().date()) if len(test_df) else None,
        "horizon": params["horizon"],
        "theta_mode": params["theta_mode"],
        "theta_k": params["theta_k"],
        "lookback_high": params["lookback_high"],
        "alpha": params["alpha"],
        "base_rate_mode": base_rate_mode,
        "n_total": len(train_df) + len(test_df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "logloss": float(log_loss(y_true, p, labels=CLASS_LABELS)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": brier_multiclass(y_true, p),
        "corr_p_growth_future_ret": float(pred_df["P_Growth"].corr(pred_df["future_ret"])),
        "corr_p_decline_future_ret": float(pred_df["P_Decline"].corr(pred_df["future_ret"])),
        "corr_p_sideways_future_ret": float(pred_df["P_Sideways"].corr(pred_df["future_ret"])),
        "corr_s_growth_future_ret": float(pred_df["S_Growth"].corr(pred_df["future_ret"])),
        "corr_s_decline_future_ret": float(pred_df["S_Decline"].corr(pred_df["future_ret"])),
        "corr_s_sideways_future_ret": float(pred_df["S_Sideways"].corr(pred_df["future_ret"])),
        "mean_margin": float(pred_df["margin"].mean()),
        "median_margin": float(pred_df["margin"].median()),
        "mean_confidence": float(pred_df["confidence"].mean()),
    }

    for cls in CLASS_LABELS:
        row[f"train_share_{CLASS_NAMES[cls]}"] = float((train_df["target_3"] == cls).mean())
        row[f"test_share_{CLASS_NAMES[cls]}"] = float((test_df["target_3"] == cls).mean())
        row[f"pred_share_{CLASS_NAMES[cls]}"] = float((pred_df["predicted_class"] == cls).mean())

    grp_pred = pred_df.groupby("predicted_class")["future_ret"].agg(["count", "mean", "median"])
    grp_true = pred_df.groupby("true_class")["future_ret"].agg(["count", "mean", "median"])
    grp_sig = pred_df.groupby("signal_leader")["future_ret"].agg(["count", "mean", "median"])

    for cls in CLASS_LABELS:
        row[f"pred_mean_future_{CLASS_NAMES[cls]}"] = float(grp_pred.loc[cls, "mean"]) if cls in grp_pred.index else np.nan
        row[f"pred_median_future_{CLASS_NAMES[cls]}"] = float(grp_pred.loc[cls, "median"]) if cls in grp_pred.index else np.nan
        row[f"true_mean_future_{CLASS_NAMES[cls]}"] = float(grp_true.loc[cls, "mean"]) if cls in grp_true.index else np.nan
        row[f"signal_mean_future_{CLASS_NAMES[cls]}"] = float(grp_sig.loc[cls, "mean"]) if cls in grp_sig.index else np.nan

    return row



def evaluate_one_run(
    ticker: str,
    period: dict[str, Any],
    params: dict[str, Any],
    *,
    market: str,
    train_frac: float,
    weights: dict[str, float],
    base_rate_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    stock_df, market_df = pt.download_two_tickers(ticker, market, period["start"], period["end"])
    ready = build_ready_df(
        stock_df,
        market_df,
        horizon=params["horizon"],
        lookback_high=params["lookback_high"],
        theta_mode=params["theta_mode"],
        theta_k=params["theta_k"],
    )

    if len(ready) < 400:
        raise ValueError(f"Too few rows after feature engineering: {len(ready)}")

    split = int(len(ready) * train_frac)
    train_df = ready.iloc[:split].copy()
    test_df = ready.iloc[split:].copy()

    if len(train_df) < 200 or len(test_df) < 50:
        raise ValueError(f"Train/test split too small. train={len(train_df)}, test={len(test_df)}")

    tables = build_tables(train_df, alpha=params["alpha"])
    empirical_base = get_base_rate(train_df["target_3"], mode="empirical")
    agg_base = get_base_rate(train_df["target_3"], mode=base_rate_mode)

    probs_dsg, score_df = predict_tables_logodds(
        test_df,
        tables,
        weights=weights,
        base_rate_dsg=agg_base,
        fallback_dsg=empirical_base,
    )

    pred_df = pd.DataFrame(
        {
            "date": test_df.index,
            "ticker": ticker,
            "P_Decline": probs_dsg[:, 0],
            "P_Sideways": probs_dsg[:, 1],
            "P_Growth": probs_dsg[:, 2],
            "predicted_class": np.argmax(probs_dsg, axis=1),
            "true_class": test_df["target_3"].astype(int).values,
            "future_ret": test_df["future_ret"].values,
            "theta": test_df["theta"].values,
            "base_D": agg_base[0],
            "base_S": agg_base[1],
            "base_G": agg_base[2],
            "period": period["name"],
            "horizon": params["horizon"],
            "theta_mode": params["theta_mode"],
            "theta_k": params["theta_k"],
            "lookback_high": params["lookback_high"],
            "alpha": params["alpha"],
        },
        index=test_df.index,
    )
    pred_df = pd.concat([pred_df, score_df], axis=1)

    pred_df["confidence"] = pred_df[["P_Decline", "P_Sideways", "P_Growth"]].max(axis=1)
    top2 = np.sort(pred_df[["P_Decline", "P_Sideways", "P_Growth"]].values, axis=1)[:, -2:]
    pred_df["margin"] = top2[:, 1] - top2[:, 0]
    pred_df["signal_leader"] = pred_df[["S_Decline", "S_Sideways", "S_Growth"]].values.argmax(axis=1)
    pred_df["signal_leader_name"] = pred_df["signal_leader"].map(CLASS_NAMES)
    pred_df["predicted_name"] = pred_df["predicted_class"].map(CLASS_NAMES)
    pred_df["true_name"] = pred_df["true_class"].map(CLASS_NAMES)
    pred_df["correct"] = (pred_df["predicted_class"] == pred_df["true_class"]).astype(int)

    summary = summarize_rows(
        pred_df,
        ticker=ticker,
        period_name=period["name"],
        params=params,
        train_df=train_df,
        test_df=test_df,
        base_rate_mode=base_rate_mode,
    )
    return summary, pred_df.reset_index(drop=True)



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    row_level_frames: list[pd.DataFrame] = []
    run_id = 0

    for ticker in TICKERS:
        for period in PERIODS:
            for params in iter_param_grid(PARAM_GRID):
                run_id += 1
                label = (
                    f"{ticker} | {period['name']} | h={params['horizon']} | "
                    f"theta={params['theta_mode']}:{params['theta_k']} | alpha={params['alpha']}"
                )
                print(f"\n[{run_id}] {label}")
                try:
                    summary, pred_df = evaluate_one_run(
                        ticker,
                        period,
                        params,
                        market=MARKET,
                        train_frac=TRAIN_FRAC,
                        weights=WEIGHTS,
                        base_rate_mode=BASE_RATE_MODE,
                    )
                    summary_rows.append(summary)
                    print(
                        f"  ✅ logloss={summary['logloss']:.4f} | "
                        f"acc={summary['accuracy']:.4f} | "
                        f"bal_acc={summary['balanced_accuracy']:.4f} | "
                        f"corr_pg={summary['corr_p_growth_future_ret']:.4f}"
                    )

                    if SAVE_ROW_LEVEL:
                        pred_df["run_id"] = run_id
                        row_level_frames.append(pred_df)
                except Exception as e:
                    print(f"  ❌ {type(e).__name__}: {e}")
                    summary_rows.append(
                        {
                            "ticker": ticker,
                            "period": period["name"],
                            **params,
                            "base_rate_mode": BASE_RATE_MODE,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / "summary_results.csv"
    summary_parquet = OUTPUT_DIR / "summary_results.parquet"
    summary_df.to_csv(summary_csv, index=False)
    try:
        summary_df.to_parquet(summary_parquet, index=False)
    except Exception:
        pass

    print("\n=== TOP RUNS BY LOGLOSS ===")
    if "logloss" in summary_df.columns:
        ok = summary_df.dropna(subset=["logloss"]).sort_values("logloss").head(20)
        print(ok[[
            "ticker", "period", "horizon", "theta_mode", "theta_k", "alpha",
            "logloss", "accuracy", "balanced_accuracy",
            "corr_p_growth_future_ret", "corr_p_decline_future_ret",
        ]].to_string(index=False))

    if SAVE_ROW_LEVEL and row_level_frames:
        rows_df = pd.concat(row_level_frames, ignore_index=True)
        rows_csv = OUTPUT_DIR / "row_level_predictions.csv"
        rows_parquet = OUTPUT_DIR / "row_level_predictions.parquet"
        rows_df.to_csv(rows_csv, index=False)
        try:
            rows_df.to_parquet(rows_parquet, index=False)
        except Exception:
            pass
        print(f"\nSaved row-level predictions: {rows_csv}")

    print(f"\nSaved summary table: {summary_csv}")


if __name__ == "__main__":
    main()
