# prob_tables Batch_runner.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import prob_tables as pt


# ============================================================================
# CONFIG: edit these lists and parameters for your experiments
# ============================================================================

TICKERS = [
    "AAPL",
    # "MSFT",
    # "GOOGL",
    # "AMD",
    # "DIS"
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
    "horizon": [30],
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

SUMMARY_CSV = "summary_results.csv"
ROW_CSV = "row_level_predictions.csv"
SUMMARY_PARQUET = "summary_results.parquet"
ROW_PARQUET = "row_level_predictions.parquet"

SUMMARY_CSV_PATH = OUTPUT_DIR / SUMMARY_CSV
ROW_CSV_PATH = OUTPUT_DIR / ROW_CSV
SUMMARY_PARQUET_PATH = OUTPUT_DIR / SUMMARY_PARQUET
ROW_PARQUET_PATH = OUTPUT_DIR / ROW_PARQUET


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


def load_row_predictions():
    return pd.read_csv(ROW_CSV_PATH)
def load_sum_result():
    return pd.read_csv(SUMMARY_CSV_PATH)
    
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

def plot_bucket_probabilities(
    tables: dict[str, pd.DataFrame],
    factor: str,
    *,
    title_prefix: str = "",
    save_path: str | None = None,
):
    """
    Bar chart of bucket probabilities for one factor.
    """
    tbl = tables[factor].copy()

    if not {"Decline", "Sideways", "Growth"}.issubset(tbl.columns):
        raise ValueError(f"Table for factor '{factor}' does not contain required columns.")

    plot_df = tbl[["Decline", "Sideways", "Growth"]].copy()

    ax = plot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_title(f"{title_prefix}Bucket probabilities: {factor}")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_bucket_heatmap(
    tables: dict[str, pd.DataFrame],
    factor: str,
    *,
    title_prefix: str = "",
    save_path: str | None = None,
):
    """
    Heatmap of bucket probabilities for one factor.
    Rows = buckets, columns = classes.
    """
    tbl = tables[factor].copy()
    plot_df = tbl[["Decline", "Sideways", "Growth"]].copy()

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(plot_df))))
    im = ax.imshow(plot_df.values, aspect="auto", interpolation="nearest")

    ax.set_title(f"{title_prefix}Bucket heatmap: {factor}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Decline", "Sideways", "Growth"])
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_probabilities_over_time(
    pred_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | None = None,
):
    d = pred_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    plt.figure(figsize=(12, 4))
    plt.plot(d["date"], d["P_Decline"], label="P_Decline")
    plt.plot(d["date"], d["P_Sideways"], label="P_Sideways")
    plt.plot(d["date"], d["P_Growth"], label="P_Growth")

    plt.title(title or "Probabilities over time")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_signal_heatmap_over_time(
    pred_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | None = None,
):
    d = pred_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    z = np.vstack([
        d["S_Decline"].values,
        d["S_Sideways"].values,
        d["S_Growth"].values,
    ])

    vmax = np.nanmax(np.abs(z))
    vmax = max(vmax, 1e-9)

    fig, ax = plt.subplots(figsize=(13, 3.8))
    im = ax.imshow(
        z,
        aspect="auto",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Decline", "Sideways", "Growth"])

    n = len(d)
    step = max(1, n // 10)
    xticks = np.arange(0, n, step)
    xlabels = d["date"].dt.strftime("%Y-%m-%d").iloc[xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    ax.set_title(title or "Signal heatmap over time")
    ax.set_xlabel("Date")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Signal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix_from_predictions(
    pred_df: pd.DataFrame,
    *,
    title: str = "",
    normalize: bool = False,
    save_path: str | None = None,
):
    y_true = pred_df["true_class"].astype(int).values
    y_pred = pred_df["predicted_class"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true" if normalize else None)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Decline", "Sideways", "Growth"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format=".2f" if normalize else "d")
    ax.set_title(title or ("Confusion matrix (%)" if normalize else "Confusion matrix"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_future_ret_by_predicted_class(
    pred_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | None = None,
):
    d = pred_df.copy()

    data = [
        d.loc[d["predicted_class"] == 0, "future_ret"].dropna().values,
        d.loc[d["predicted_class"] == 1, "future_ret"].dropna().values,
        d.loc[d["predicted_class"] == 2, "future_ret"].dropna().values,
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, tick_labels=["Decline", "Sideways", "Growth"])
    plt.axhline(0, linestyle="--", alpha=0.5)

    plt.title(title or "Future return by predicted class")
    plt.ylabel("future_ret")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_growth_deciles(
    pred_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | None = None,
):

    d = pred_df.copy()

    d["decile"] = pd.qcut(
        d["P_Growth"],
        10,
        labels=False,
        duplicates="drop"
    )

    grp = d.groupby("decile")["future_ret"].mean()

    plt.figure(figsize=(7,5))
    plt.bar(grp.index, grp.values)

    plt.title(title or "Future return by P_Growth decile")
    plt.xlabel("P_Growth decile (0 = lowest)")
    plt.ylabel("Mean future_ret")

    plt.axhline(0, linestyle="--")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_strategy_curve(
    pred_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | None = None,
):

    d = pred_df.copy()
    d = d.sort_values("date")

    signal = np.where(
        d["predicted_class"] == 2,
        1,
        np.where(
            d["predicted_class"] == 0,
            -1,
            0
        )
    )

    strat_ret = signal * d["future_ret"]

    equity = (1 + strat_ret).cumprod()

    plt.figure(figsize=(10,4))

    plt.plot(d["date"], equity)

    plt.title(title or "Strategy equity curve")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

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

def plot_one_run_dashboard(
    ticker: str,
    period: dict[str, Any],
    params: dict[str, Any],
    *,
    market: str,
    train_frac: float,
    weights: dict[str, float],
    base_rate_mode: str,
    output_dir: Path | None = None,
):
    """
    Re-runs one experiment and draws charts.
    """
    stock_df, market_df = pt.download_two_tickers(ticker, market, period["start"], period["end"])
    ready = build_ready_df(
        stock_df,
        market_df,
        horizon=params["horizon"],
        lookback_high=params["lookback_high"],
        theta_mode=params["theta_mode"],
        theta_k=params["theta_k"],
    )

    split = int(len(ready) * train_frac)
    train_df = ready.iloc[:split].copy()
    test_df = ready.iloc[split:].copy()

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
        },
        index=test_df.index,
    )
    pred_df = pd.concat([pred_df, score_df], axis=1)
    pred_df["confidence"] = pred_df[["P_Decline", "P_Sideways", "P_Growth"]].max(axis=1)

    top2 = np.sort(pred_df[["P_Decline", "P_Sideways", "P_Growth"]].values, axis=1)[:, -2:]
    pred_df["margin"] = top2[:, 1] - top2[:, 0]
    pred_df["signal_leader"] = pred_df[["S_Decline", "S_Sideways", "S_Growth"]].values.argmax(axis=1)

    prefix = f"{ticker} | {period['name']} | h={params['horizon']} | {params['theta_mode']} | theta_k={params['theta_k']}"

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 3 factor bucket charts
    for factor in ["rsi", "volpct", "dist"]:
        plot_bucket_probabilities(
            tables,
            factor,
            title_prefix=prefix + " | ",
            save_path=str(output_dir / f"{ticker}_{period['name']}_{factor}_bars.png") if output_dir else None,
        )
        plot_bucket_heatmap(
            tables,
            factor,
            title_prefix=prefix + " | ",
            save_path=str(output_dir / f"{ticker}_{period['name']}_{factor}_heatmap.png") if output_dir else None,
        )

    plot_probabilities_over_time(
        pred_df,
        title=prefix + " | Probabilities over time",
        save_path=str(output_dir / f"{ticker}_{period['name']}_probabilities.png") if output_dir else None,
    )

    plot_signal_heatmap_over_time(
        pred_df,
        title=prefix + " | Signal heatmap",
        save_path=str(output_dir / f"{ticker}_{period['name']}_signal_heatmap.png") if output_dir else None,
    )

    plot_confusion_matrix_from_predictions(
        pred_df,
        title=prefix + " | Confusion matrix",
        normalize=False,
        save_path=str(output_dir / f"{ticker}_{period['name']}_confusion_counts.png") if output_dir else None,
    )

    plot_confusion_matrix_from_predictions(
        pred_df,
        title=prefix + " | Confusion matrix (%)",
        normalize=True,
        save_path=str(output_dir / f"{ticker}_{period['name']}_confusion_pct.png") if output_dir else None,
    )

    plot_future_ret_by_predicted_class(
        pred_df,
        title=prefix + " | future_ret by predicted class",
        save_path=str(output_dir / f"{ticker}_{period['name']}_future_ret_by_pred_class.png") if output_dir else None,
    )

    plot_growth_deciles(
        pred_df,
        title=prefix + " | P_Growth deciles",
        save_path=str(output_dir / f"{ticker}_{period['name']}_growth_deciles.png") if output_dir else None,
    )

    plot_strategy_curve(
        pred_df,
        title=prefix + " | Strategy curve",
        save_path=str(output_dir / f"{ticker}_{period['name']}_strategy_curve.png") if output_dir else None,
    )

    return pred_df, tables

def load_best_run_from_csv(
    csv_path: str | Path,
    *,
    sort_by: str = "logloss",
    ascending: bool = True,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """
    Reads summary CSV and returns:
      best_ticker, best_period, best_params
    """

    df = load_sum_result()

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    df = df.dropna(subset=[sort_by]).copy()

    if len(df) == 0:
        raise ValueError("No valid rows found in summary CSV.")

    best = df.sort_values(sort_by, ascending=ascending).iloc[0]

    best_ticker = best["ticker"]

    best_period = next(
        p for p in PERIODS if p["name"] == best["period"]
    )

    best_params = {
        "horizon": int(best["horizon"]),
        "theta_mode": best["theta_mode"],
        "theta_k": float(best["theta_k"]),
        "lookback_high": int(best["lookback_high"]),
        "alpha": float(best["alpha"]),
    }

    return best_ticker, best_period, best_params

def plot_best_run_from_csv(
    csv_path: str | Path,
    *,
    sort_by: str = "logloss",
    ascending: bool = True,
    output_subdir: str = "best_run_charts_from_csv",
):
    best_ticker, best_period, best_params = load_best_run_from_csv(
        csv_path,
        sort_by=sort_by,
        ascending=ascending,
    )

    print("\nBEST RUN FROM CSV:")
    print("ticker:", best_ticker)
    print("period:", best_period)
    print("params:", best_params)

    return plot_one_run_dashboard(
        ticker=best_ticker,
        period=best_period,
        params=best_params,
        market=MARKET,
        train_frac=TRAIN_FRAC,
        weights=WEIGHTS,
        base_rate_mode=BASE_RATE_MODE,
        output_dir=OUTPUT_DIR / output_subdir,
    )


def list_runs_for_ticker(
    csv_path: str | Path,
    *,
    ticker: str,
    period_name: str | None = None,
    sort_by: str = "logloss",
    ascending: bool = True,
    top_n: int = 20,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read summary CSV and return runs for one ticker, optionally one period.

    Parameters
    ----------
    csv_path : SUMMARY_CSV_PATH
    ticker : e.g. "DIS"
    period_name : optional, e.g. "2020_now"
    sort_by : any column from summary CSV, e.g.
              "logloss", "accuracy", "balanced_accuracy",
              "corr_p_growth_future_ret", "corr_p_decline_future_ret"
    ascending : True for lower-is-better metrics like logloss,
                False for higher-is-better metrics like accuracy/correlation
    top_n : number of rows to return
    columns : optional list of columns to display/return

    Returns
    -------
    pd.DataFrame
    """

    df = load_sum_result()



    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    df = df[df["ticker"] == ticker].copy()

    if period_name is not None:
        df = df[df["period"] == period_name].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for ticker={ticker}, period={period_name}")

    if sort_by not in df.columns:
        raise ValueError(f"Column '{sort_by}' not found in CSV. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[sort_by]).copy()
    df = df.sort_values(sort_by, ascending=ascending).head(top_n)

    if columns is None:
        default_cols = [
            "ticker", "period", "horizon", "theta_mode", "theta_k", "alpha", "lookback_high",
            "logloss", "accuracy", "balanced_accuracy",
            "corr_p_growth_future_ret", "corr_p_decline_future_ret",
            "corr_p_sideways_future_ret",
            "mean_confidence", "mean_margin"
        ]
        columns = [c for c in default_cols if c in df.columns]

    return df[columns].reset_index(drop=True)

def list_runs_all_tickers(
    csv_path: str | Path,
    *,
    period_name: str | None = None,
    sort_by: str = "logloss",
    ascending: bool = True,
    top_n: int = 50,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    List runs across all tickers.

    Parameters
    ----------
    csv_path : SUMMARY_CSV_PATH
    period_name : optional filter
    sort_by : column to sort by
    ascending : True for metrics like logloss
    top_n : number of rows
    columns : optional subset of columns
    """

    df = load_sum_result()

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    if period_name is not None:
        df = df[df["period"] == period_name].copy()

    if sort_by not in df.columns:
        raise ValueError(f"{sort_by} not in dataframe columns")

    df = df.dropna(subset=[sort_by])
    df = df.sort_values(sort_by, ascending=ascending).head(top_n)

    if columns is None:
        columns = [
            "ticker",
            "period",
            "horizon",
            "theta_mode",
            "theta_k",
            "alpha",
            "logloss",
            "accuracy",
            "balanced_accuracy",
            "corr_p_growth_future_ret",
            "corr_p_decline_future_ret",
        ]

    columns = [c for c in columns if c in df.columns]

    return df[columns].reset_index(drop=True)

def best_run_per_ticker(
    csv_path: str | Path,
    *,
    period_name: str | None = None,
    sort_by: str = "logloss",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Returns best run per ticker.
    """

    df = load_sum_result()

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    if period_name is not None:
        df = df[df["period"] == period_name]

    df = df.dropna(subset=[sort_by])

    df = df.sort_values(sort_by, ascending=ascending)

    best = df.groupby("ticker").head(1)

    cols = [
        "ticker",
        "period",
        "horizon",
        "theta_mode",
        "theta_k",
        "logloss",
        "accuracy",
        "balanced_accuracy",
        "corr_p_growth_future_ret",
        "corr_p_decline_future_ret",
    ]

    cols = [c for c in cols if c in best.columns]

    return best[cols].reset_index(drop=True)

def plot_metric_by_ticker(
    csv_path: str | Path,
    *,
    metric: str = "logloss",
    period_name: str | None = None,
    ascending: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Plot best value of chosen metric for each ticker.

    Parameters
    ----------
    csv_path : SUMMARY_CSV_PATH
    metric : metric column, e.g.
             'logloss', 'accuracy', 'balanced_accuracy',
             'corr_p_growth_future_ret', 'corr_p_decline_future_ret'
    period_name : optional filter, e.g. '2020_now'
    ascending : True if lower metric is better (logloss, brier),
                False if higher metric is better (accuracy, correlation)
    title : optional custom title
    save_path : optional file path to save figure
    """

    best = best_run_per_ticker(
        csv_path,
        period_name=period_name,
        sort_by=metric,
        ascending=ascending,
    ).copy()

    if metric not in best.columns:
        raise ValueError(f"Metric '{metric}' not found in best_run_per_ticker output.")

    best = best.sort_values(metric, ascending=ascending)

    plt.figure(figsize=(9, 5))
    plt.bar(best["ticker"], best[metric])

    plt.title(title or f"Best {metric} by ticker" + (f" | {period_name}" if period_name else ""))
    plt.xlabel("Ticker")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


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
    summary_csv = SUMMARY_CSV_PATH
    summary_parquet = SUMMARY_PARQUET_PATH
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
        rows_csv = ROW_CSV_PATH
        rows_parquet = ROW_PARQUET_PATH
        rows_df.to_csv(rows_csv, index=False)
        try:
            rows_df.to_parquet(rows_parquet, index=False)
        except Exception:
            pass
        print(f"\nSaved row-level predictions: {rows_csv}")

    print(f"\nSaved summary table: {summary_csv}")

def mean_run_per_ticker(
    csv_path: str | Path,
    *,
    period_name: str | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns mean metric values per ticker across all runs.
    """

    df = load_sum_result()

    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    if period_name is not None:
        df = df[df["period"] == period_name].copy()

    if metrics is None:
        metrics = [
            "logloss",
            "accuracy",
            "balanced_accuracy",
            "brier",
            "corr_p_growth_future_ret",
            "corr_p_decline_future_ret",
            "corr_p_sideways_future_ret",
            "mean_confidence",
            "mean_margin",
        ]

    metrics = [m for m in metrics if m in df.columns]

    if len(metrics) == 0:
        raise ValueError("No requested metric columns found in CSV.")

    out = df.groupby("ticker")[metrics].mean().reset_index()
    return out

def plot_multi_metric_by_ticker_best(
    csv_path: str | Path,
    *,
    metrics: list[str],
    best_by: str = "logloss",
    best_by_ascending: bool = True,
    period_name: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    For each ticker:
      choose best run by `best_by`,
      then plot several metrics for that best run.
    """

    best = best_run_per_ticker(
        csv_path,
        period_name=period_name,
        sort_by=best_by,
        ascending=best_by_ascending,
    ).copy()

    metrics = [m for m in metrics if m in best.columns]

    if len(metrics) == 0:
        raise ValueError("None of the requested metrics are present.")

    plot_df = best.set_index("ticker")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title or f"Best run per ticker | selected by {best_by}" + (f" | {period_name}" if period_name else ""))
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_multi_metric_by_ticker_mean(
    csv_path: str | Path,
    *,
    metrics: list[str],
    period_name: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    For each ticker:
      take mean metric across all runs,
      then plot several metrics.
    """

    mean_df = mean_run_per_ticker(
        csv_path,
        period_name=period_name,
        metrics=metrics,
    ).copy()

    metrics = [m for m in metrics if m in mean_df.columns]

    if len(metrics) == 0:
        raise ValueError("None of the requested metrics are present.")

    plot_df = mean_df.set_index("ticker")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title or "Mean metrics per ticker" + (f" | {period_name}" if period_name else ""))
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def table_multi_metric_by_ticker_best(
    csv_path: str | Path,
    *,
    metrics: list[str],
    best_by: str = "logloss",
    best_by_ascending: bool = True,
    period_name: str | None = None,
) -> pd.DataFrame:
    best = best_run_per_ticker(
        csv_path,
        period_name=period_name,
        sort_by=best_by,
        ascending=best_by_ascending,
    ).copy()

    cols = ["ticker"] + [m for m in metrics if m in best.columns]
    return best[cols].reset_index(drop=True)

def table_multi_metric_by_ticker_mean(
    csv_path: str | Path,
    *,
    metrics: list[str],
    period_name: str | None = None,
) -> pd.DataFrame:
    mean_df = mean_run_per_ticker(
        csv_path,
        period_name=period_name,
        metrics=metrics,
    ).copy()

    cols = ["ticker"] + [m for m in metrics if m in mean_df.columns]
    return mean_df[cols].reset_index(drop=True)




def plot_probability_calibration(
    pred_df: pd.DataFrame,
    *,
    prob_col: str = "P_Growth",
    target_col: str = "future_ret",
    bins: int = 10,
    title: str = "Probability calibration",
    save_path: str | None = None,
):
    """
    Plot probability buckets vs realized future return.
    """

    df = pred_df.copy()

    df["prob_bucket"] = pd.cut(df[prob_col], bins=bins)

    stats = df.groupby("prob_bucket")[target_col].agg(
        count="count",
        mean="mean",
        median="median"
    ).reset_index()

    centers = df.groupby("prob_bucket")[prob_col].mean().values

    plt.figure(figsize=(8,5))

    plt.plot(
        centers,
        stats["mean"],
        marker="o",
        linewidth=2
    )

    plt.axhline(0, linestyle="--", alpha=0.5)

    plt.xlabel(prob_col)
    plt.ylabel("mean future return")

    plt.title(title)

    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    return stats

def plot_probability_vs_frequency(
    pred_df,
    *,
    prob_col="P_Growth",
    true_col="true_class",
    bins=10,
):
    df = pred_df.copy()

    df["prob_bucket"] = pd.cut(df[prob_col], bins=bins)

    df["is_growth"] = (df[true_col] == 2).astype(int)

    stats = df.groupby("prob_bucket").agg(
        mean_prob=(prob_col, "mean"),
        growth_freq=("is_growth", "mean"),
        count=("is_growth", "count"),
    )

    plt.figure(figsize=(6,6))

    plt.plot(stats["mean_prob"], stats["growth_freq"], marker="o")

    plt.plot([0,1],[0,1], linestyle="--", alpha=0.5)

    plt.xlabel("Predicted probability")
    plt.ylabel("Real frequency")

    plt.title("Probability calibration")

    plt.grid(alpha=0.3)

    plt.show()

    return stats

def plot_decile_return(
    pred_df: pd.DataFrame,
    *,
    prob_col: str = "P_Growth",
    target_col: str = "future_ret",
    n_bins: int = 10,
    title: str = "Decile return plot",
    save_path: str | None = None,
):
    df = pred_df.copy()

    df = df[[prob_col, target_col]].dropna().copy()
    df["decile"] = pd.qcut(df[prob_col], q=n_bins, labels=False, duplicates="drop")

    stats = df.groupby("decile")[target_col].agg(["count", "mean", "median"]).reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(stats["decile"].astype(str), stats["mean"])
    plt.axhline(0, linestyle="--", alpha=0.5)

    plt.title(title)
    plt.xlabel(f"{prob_col} quantile bucket")
    plt.ylabel(f"Mean {target_col}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return stats

if __name__ == "__main__":
    main()

    # runs = list_runs_all_tickers(
    #     SUMMARY_CSV_PATH,
    #     sort_by="corr_p_growth_future_ret",
    #     ascending=False,
    #     top_n=30,
    # )

    # print(runs.to_string(index=False))

    # runs = list_runs_for_ticker(
    #     SUMMARY_CSV_PATH",
    #     ticker="GOOGL",
    #     sort_by="logloss", #   accuracy  balanced_accuracy  corr_p_growth_future_ret  corr_p_decline_future_ret  corr_p_sideways_future_ret  mean_confidence  mean_margin
    #     ascending=True,  # False, якщо більша метрика краще.
    #     top_n=30,
    # )

    # print(runs.to_string(index=False))

    # plot_best_run_from_csv(
    #     SUMMARY_CSV_PATH",
    #     sort_by="logloss",  # sort_by="corr_p_growth_future_ret", "balanced_accuracy"
    #     ascending=True,
    #     output_subdir="best_run_charts_from_csv",
    # )

# pred_df = load_row_predictions()
# )

# df = load_sum_result()

# cols = [
#     "ticker",
#     "period",
#     "horizon",
#     "pred_mean_future_Growth",
#     "pred_mean_future_Sideways",
#     "pred_mean_future_Decline",
#     "signal_mean_future_Growth",
#     "signal_mean_future_Decline",
#     "signal_mean_future_Sideways"

# ]

# print(df[cols].sort_values("pred_mean_future_Growth", ascending=False).head(20))

# pred_df = load_row_predictions()

# df = pred_df[pred_df["ticker"] == "AAPL"]

# plot_probability_calibration(
#     df,
#     prob_col="P_Growth",
#     target_col="future_ret",
#     bins=10,
#     title="AAPL probability calibration"
# )

# # probability vs frequency
# plot_probability_vs_frequency(
#     df,
#     prob_col="P_Growth",
#     true_col="true_class",
#     bins=10
# )

# plot_decile_return(
#     df,
#     prob_col="P_Growth",
#     target_col="future_ret",
#     n_bins=10,
#     title="AAPL decile return plot"
# )

