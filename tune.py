# tune.py  - Optuna
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from core import download_two_tickers, build_ready_df, build_tables_from_train_df, predict_tables_for_df, gds_to_dsg
# -------- helpers --------
def gds_to_dsg(p_gds):
    return p_gds[:, [1,2,0]]  # [G,D,S] -> [D,S,G]

def safe_clip_probs(p, eps=1e-9):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def make_walkforward_folds(index, n_folds=5, min_train=0.5):
    n = len(index)
    start_train = int(n * min_train)
    fold_sizes = (n - start_train) // n_folds
    folds = []
    for k in range(n_folds):
        train_end = start_train + k * fold_sizes
        val_end = start_train + (k + 1) * fold_sizes if k < n_folds - 1 else n
        if train_end < 200 or (val_end - train_end) < 100:
            continue
        folds.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    return folds

def softmax_logits(logits_dict):
    keys = ["dist","rsi","macd","vol","trend","market"]
    z = np.array([logits_dict[k] for k in keys], dtype=float)
    z = z - z.max()  # stability
    w = np.exp(z)
    w = w / w.sum()
    return dict(zip(keys, w))

# -------- core scoring: walk-forward logloss on train --------
def score_tables_walkforward(train_df, params, weights_6, n_folds=5):
    folds = make_walkforward_folds(train_df.index, n_folds=n_folds, min_train=0.5)
    if len(folds) < 2:
        return np.inf

    y = train_df["target_3"].astype(int).values
    probs_all = []
    y_all = []

    for tr_pos, va_pos in folds:
        tr = train_df.iloc[tr_pos].copy()
        va = train_df.iloc[va_pos].copy()

        fb = tr["target_3"].value_counts(normalize=True).reindex([2,0,1]).fillna(1/3).values
        tables = build_tables_from_train_df(tr, alpha=params["alpha"])
        p_va_gds = predict_tables_for_df(va, tables, weights_6, fb)

        probs_all.append(gds_to_dsg(p_va_gds))
        y_all.append(va["target_3"].astype(int).values)

    P = np.vstack(probs_all)
    Y = np.concatenate(y_all)
    P = safe_clip_probs(P)
    return log_loss(Y, P, labels=[0,1,2])

# -------- Optuna tuner --------
def tune_one_ticker_optuna(
    stock_ticker,
    market_ticker="SPY",
    start_date="2010-01-01",
    end_date=None,
    train_frac=0.8,
    n_trials=80,
    seed=42
):
    # Prepare full df once (but params will rebuild target/rolling high, so we rebuild inside objective)
    s, m = download_two_tickers(stock_ticker, market_ticker, start_date, end_date)

    def objective(trial):
        # ---- params to tune ----
        horizon = trial.suggest_int("horizon", 20, 180, step=20)
        theta_mode = trial.suggest_categorical("theta_mode", ["vol","atr"])
        theta_k = trial.suggest_float("theta_k", 0.10, 0.80)
        lookback_high = trial.suggest_categorical("lookback_high", [126, 252])
        alpha = trial.suggest_float("alpha", 0.2, 5.0)

        params = {
            "horizon": horizon,
            "theta_mode": theta_mode,
            "theta_k": theta_k,
            "lookback_high": lookback_high,
            "alpha": alpha
        }

        # rebuild df with these params (target depends on horizon/theta)
        d_all = build_ready_df(
            s, m,
            horizon=horizon,
            lookback_high=lookback_high,
            theta_mode=theta_mode,
            theta_k=theta_k
        )

        split = int(len(d_all)*train_frac)
        train = d_all.iloc[:split].copy()

        # ---- weights logits to tune ----
        logits = {
            "dist":  trial.suggest_float("w_dist",  -3.0, 3.0),
            "rsi":   trial.suggest_float("w_rsi",   -3.0, 3.0),
            "macd":  trial.suggest_float("w_macd",  -3.0, 3.0),
            "vol":   trial.suggest_float("w_vol",   -3.0, 3.0),
            "trend": trial.suggest_float("w_trend", -3.0, 3.0),
            "market":trial.suggest_float("w_market",-3.0, 3.0),
        }
        weights_6 = softmax_logits(logits)

        ll = score_tables_walkforward(train, params, weights_6, n_folds=5)
        return ll  # minimize

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    ## ------------------------
    best = study.best_trial
    print("BEST trial #:", best.number)
    print("BEST CV logloss:", best.value)
    print("BEST params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    # -------------------------

    best = study.best_params
    best_params = {
        "horizon": best["horizon"],
        "theta_mode": best["theta_mode"],
        "theta_k": best["theta_k"],
        "lookback_high": best["lookback_high"],
        "alpha": best["alpha"],
    }
    best_logits = {
        "dist": best["w_dist"],
        "rsi": best["w_rsi"],
        "macd": best["w_macd"],
        "vol": best["w_vol"],
        "trend": best["w_trend"],
        "market": best["w_market"],
    }
    best_weights_6 = softmax_logits(best_logits)

    return study, best_params, best_weights_6
