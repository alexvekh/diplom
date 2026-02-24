# core.py
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from prob_tables import (download_two_tickers, 
            create_stock_features, create_market_features, make_target_3,
            bucket_distance_from_high, bucket_rsi, macd_status, trend_bucket,
            bucket_volume, market_bucket, prob_table_3class)

# ===========  PROBABILITY TABLES PIPELINE  ===============

def build_prediction_table_for_ticker(
    stock_ticker,
    market_ticker,
    start_date,
    end_date,
    params,
    weights_6,
    train_frac=0.8
):
    stock, market = download_two_tickers(
        stock_ticker, market_ticker, start_date, end_date
    )

    d_all = build_ready_df(
        stock, market,
        horizon=params["horizon"],
        theta_mode=params["theta_mode"],
        theta_k=params["theta_k"],
        lookback_high=params["lookback_high"]
    )

    split = int(len(d_all) * train_frac)
    train = d_all.iloc[:split].copy()
    test  = d_all.iloc[split:].copy()

    fallback_gds = train["target_3"].value_counts(normalize=True)\
        .reindex([2,0,1]).fillna(1/3).values

    tables = build_tables_from_train_df(train, alpha=params["alpha"])

    # Tables probabilities
    p_tab_gds = predict_tables_for_df(test, tables, weights_6, fallback_gds)

    # convert to [Decline, Sideways, Growth]
    proba_dsg = gds_to_dsg(p_tab_gds)

    y_true = test["target_3"].astype(int).values
    y_pred = np.argmax(proba_dsg, axis=1)

    pred_df = pd.DataFrame({
        "date": test.index,
        "P_Decline": proba_dsg[:,0],
        "P_Sideways": proba_dsg[:,1],
        "P_Growth": proba_dsg[:,2],
        "predicted_class": y_pred,
        "true_class": y_true,
        "correct": (y_pred == y_true).astype(int)
    })

    return pred_df

def brier_multiclass(y_true, proba_dsg, labels=(0,1,2)):
    """
    Multiclass Brier score:
    mean over samples of sum_k (p_k - y_k)^2
    proba_dsg columns must correspond to labels order [0,1,2]
    """
    y_true = np.asarray(y_true, dtype=int)
    P = np.asarray(proba_dsg, dtype=float)
    K = len(labels)
    Y = np.zeros((len(y_true), K), dtype=float)
    label_to_col = {lab:i for i,lab in enumerate(labels)}
    for i, yt in enumerate(y_true):
        Y[i, label_to_col[yt]] = 1.0
    return np.mean(np.sum((P - Y) ** 2, axis=1))

def eval_one_ticker_tables_meta(
    stock_ticker,
    market_ticker,
    start_date="2010-01-01",
    end_date=None,
    params=None,
    weights_6=None,
    train_frac=0.8,
    seed=42
):
    """
    Returns dict with metrics for:
      - Tables_only
      - Tables→Meta-LogReg (robust OOF)
    """
    # 1) Data + features/target
    stock, market = download_two_tickers(stock_ticker, market_ticker, start_date, end_date)
    d_all = build_ready_df(
        stock, market,
        horizon=params["horizon"],
        theta_mode=params["theta_mode"],
        theta_k=params["theta_k"],
        lookback_high=params["lookback_high"]
    )

    # time split
    split = int(len(d_all) * train_frac)
    train = d_all.iloc[:split].copy()
    test  = d_all.iloc[split:].copy()

    # fallback distribution in [G,D,S] order (як у вашій таблиці)
    fallback_gds = train["target_3"].value_counts(normalize=True).reindex([2,0,1]).fillna(1/3).values

    # 2) Train tables on full train
    tables = build_tables_from_train_df(train, alpha=params["alpha"])

    # 3) Tables-only predictions
    y_true = test["target_3"].astype(int).values
    p_tab_gds = predict_tables_for_df(test, tables, weights_6, fallback_gds)  # [G,D,S]
    proba_tab_dsg = gds_to_dsg(p_tab_gds)  # -> columns [D,S,G] for labels [0,1,2]

    ll_tab = log_loss(y_true, proba_tab_dsg, labels=[0,1,2])
    acc_tab = accuracy_score(y_true, np.argmax(proba_tab_dsg, axis=1))
    br_tab = brier_multiclass(y_true, proba_tab_dsg, labels=(0,1,2))

    # 4) Robust meta training via OOF on train
    meta, diag = train_meta_with_oof_tables(
        train_df=train,
        weights_6=weights_6,
        alpha=params["alpha"],
        n_folds=5,
        seed=seed
    )

    # 5) Meta predictions
    p_meta_gds = predict_meta_from_tables(meta, test, tables, weights_6, fallback_gds)  # [G,D,S]
    proba_meta_dsg = gds_to_dsg(p_meta_gds)

    ll_meta = log_loss(y_true, proba_meta_dsg, labels=[0,1,2])
    acc_meta = accuracy_score(y_true, np.argmax(proba_meta_dsg, axis=1))
    br_meta = brier_multiclass(y_true, proba_meta_dsg, labels=(0,1,2))

    return {
        "ticker": stock_ticker,
        "n_total": len(d_all),
        "n_train": len(train),
        "n_test": len(test),

        "tables_logloss": ll_tab,
        "tables_accuracy": acc_tab,
        "tables_brier": br_tab,

        "meta_logloss": ll_meta,
        "meta_accuracy": acc_meta,
        "meta_brier": br_meta,

        "delta_logloss": ll_meta - ll_tab,

        # діагностика OOF (корисно для диплому)
        "oof_tables_logloss": diag["ll_tables_oof"],
        "oof_meta_logloss": diag["ll_meta_oof"],
        "oof_n": diag["n_oof"],
    }

def eval_many_tickers(
    tickers,
    market_ticker="SPY",
    start_date="2010-01-01",
    end_date=None,
    params=None,
    weights_6=None,
    train_frac=0.8,
    seed=42
):
    rows = []
    errors = []

    for t in tickers:
        try:
            r = eval_one_ticker_tables_meta(
                stock_ticker=t,
                market_ticker=market_ticker,
                start_date=start_date,
                end_date=end_date,
                params=params,
                weights_6=weights_6,
                train_frac=train_frac,
                seed=seed
            )
            rows.append(r)
            print(f"✅ {t}: tables={r['tables_logloss']:.3f} meta={r['meta_logloss']:.3f} Δ={r['delta_logloss']:+.3f}")
        except Exception as e:
            errors.append((t, str(e)))
            print(f"❌ {t}: {e}")

    df_res = pd.DataFrame(rows).sort_values("meta_logloss")
    df_err = pd.DataFrame(errors, columns=["ticker","error"])

    # summary
    if len(df_res) > 0:
        win_rate = (df_res["delta_logloss"] < 0).mean()
        print("\n=== SUMMARY ===")
        print(f"Tickers evaluated: {len(df_res)}/{len(tickers)}")
        print(f"Meta improves (Δ<0): {win_rate*100:.1f}% tickers")
        print(f"Median Tables logloss: {df_res['tables_logloss'].median():.3f}")
        print(f"Median Meta   logloss: {df_res['meta_logloss'].median():.3f}")
        print(f"Median Δ logloss:      {df_res['delta_logloss'].median():+.3f}")

    return df_res, df_err

def plot_multi_ticker_results(df_res, title_prefix="Across tickers"):
    if df_res is None or len(df_res) == 0:
        print("No results to plot.")
        return

    # 1) Boxplot: logloss
    plt.figure(figsize=(7,4))
    plt.boxplot([df_res["tables_logloss"].values, df_res["meta_logloss"].values],
                labels=["Tables", "Tables → Meta"])
    plt.ylabel("LogLoss")
    plt.title(f"{title_prefix}: LogLoss distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) Histogram: delta
    plt.figure(figsize=(7,4))
    plt.hist(df_res["delta_logloss"].values, bins=20, alpha=0.8)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Δ LogLoss (meta - tables)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: Δ LogLoss (negative = better)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3) Scatter: tables vs meta
    plt.figure(figsize=(6,5))
    plt.scatter(df_res["tables_logloss"], df_res["meta_logloss"], alpha=0.8)
    lim_min = min(df_res["tables_logloss"].min(), df_res["meta_logloss"].min())
    lim_max = max(df_res["tables_logloss"].max(), df_res["meta_logloss"].max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlabel("Tables logloss")
    plt.ylabel("Meta logloss")
    plt.title(f"{title_prefix}: Tables vs Meta")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def build_predictions_many_tickers(
    tickers,
    market_ticker,
    start_date,
    end_date,
    params,
    weights_6,
    train_frac=0.8
):
    all_preds = []

    for t in tickers:
        try:
            df_pred = build_prediction_table_for_ticker(
                stock_ticker=t,
                market_ticker=market_ticker,
                start_date=start_date,
                end_date=end_date,
                params=params,
                weights_6=weights_6,
                train_frac=train_frac
            )
            df_pred["ticker"] = t
            all_preds.append(df_pred)
            print(f"✅ {t} done")
        except Exception as e:
            print(f"❌ {t} error: {e}")

    return pd.concat(all_preds, ignore_index=True)

def build_ready_df(stock_df: pd.DataFrame, market_df: pd.DataFrame,
                   horizon: int, lookback_high: int,
                   theta_mode: str, theta_k: float) -> pd.DataFrame:
    s = create_stock_features(stock_df)
    m = create_market_features(market_df)

    s = s.join(m[["spy_ret","spy_trend_regime","spy_dd_pct","spy_volatility"]], how="left")

    # future return
    s["future_ret"] = s["close"].shift(-horizon) / s["close"] - 1

    # theta band
    if theta_mode == "atr":
        s["theta"] = theta_k * s["atr_pct"]
    else:
        s["theta"] = theta_k * s["volatility_20"]

    s["target_3"] = make_target_3(s["future_ret"], s["theta"])

    # distance from rolling high
    roll_high = s["close"].rolling(lookback_high).max()
    s["dd_pct"] = (roll_high - s["close"]) / roll_high * 100

    # buckets
    s["dist_bucket"] = bucket_distance_from_high(s["dd_pct"])
    s["rsi_bucket"] = bucket_rsi(s["rsi"])
    s["macd_bucket"] = macd_status(s["close"])
    s["trend_bucket"] = trend_bucket(s["close"], s["ma200"], s["ma50"])
    s["vol_bucket"] = bucket_volume(s["vol_ratio"], s["close"], s["open"])
    s["market_bucket"] = market_bucket(s["spy_trend_regime"], s["spy_dd_pct"])

    return s.dropna().copy()

def build_tables_from_train_df(train_df: pd.DataFrame, alpha: float = 1.0) -> dict:
    return {
        "dist":  prob_table_3class(train_df["dist_bucket"], train_df["target_3"], alpha=alpha),
        "rsi":   prob_table_3class(train_df["rsi_bucket"], train_df["target_3"], alpha=alpha),
        "macd":  prob_table_3class(train_df["macd_bucket"], train_df["target_3"], alpha=alpha),
        "vol":   prob_table_3class(train_df["vol_bucket"], train_df["target_3"], alpha=alpha),
        "trend": prob_table_3class(train_df["trend_bucket"], train_df["target_3"], alpha=alpha),
        "market":prob_table_3class(train_df["market_bucket"], train_df["target_3"], alpha=alpha),
    }

def predict_tables_for_df(d_df: pd.DataFrame, tables: dict, weights_6: dict, fallback_gds: np.ndarray) -> np.ndarray:
    """
    Returns p_tables in [G,D,S] order for each row of d_df.
    weights_6 keys: dist,rsi,macd,vol,trend,market
    fallback_gds: array [P(G),P(D),P(S)] used when bucket not found
    """
    W = np.array([weights_6[k] for k in ["dist","rsi","macd","vol","trend","market"]], dtype=float)

    out = np.zeros((len(d_df), 3), dtype=float)  # [G,D,S]
    for i, (_, row) in enumerate(d_df.iterrows()):
        comps = []
        for name, col in [("dist","dist_bucket"),("rsi","rsi_bucket"),("macd","macd_bucket"),
                          ("vol","vol_bucket"),("trend","trend_bucket"),("market","market_bucket")]:
            tbl = tables[name]
            b = row[col]
            if b in tbl.index:
                r = tbl.loc[b][["Growth","Decline","Sideways"]].values.astype(float)
            else:
                r = np.array(fallback_gds, dtype=float)
            comps.append(r)

        mat = np.vstack(comps)  # (6,3)
        p = np.average(mat, axis=0, weights=W)
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()
        out[i] = p

    return out



def safe_clip_probs(p, eps=1e-9):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p

# def gds_to_dsg(p_gds):
#     return p_gds[:, [1, 2, 0]]  # [G,D,S] -> [D,S,G]

def gds_to_dsg(p_gds: np.ndarray) -> np.ndarray:
    """
    Преобразование [Growth, Decline, Sideways] -> [Decline, Sideways, Growth]
    """
    p_gds = np.asarray(p_gds, dtype=float)
    return p_gds[:, [1, 2, 0]]

def probs_to_logits(p_gds, eps=1e-9):
    p = safe_clip_probs(p_gds, eps=eps)
    PG, PD, PS = p[:, 0], p[:, 1], p[:, 2]
    z1 = np.log(PG / PS)
    z2 = np.log(PD / PS)
    return np.vstack([z1, z2]).T

def brier_multiclass(y_true, proba_dsg, labels=(0,1,2)):
    y_true = np.asarray(y_true, dtype=int)
    P = np.asarray(proba_dsg, dtype=float)
    K = len(labels)
    Y = np.zeros((len(y_true), K), dtype=float)
    label_to_col = {lab:i for i,lab in enumerate(labels)}
    for i, yt in enumerate(y_true):
        Y[i, label_to_col[yt]] = 1.0
    return np.mean(np.sum((P - Y) ** 2, axis=1))

def make_walkforward_folds(index, n_folds=5, min_train=0.5):
    n = len(index)
    start_train = int(n * min_train)
    fold_sizes = (n - start_train) // n_folds
    folds = []
    for k in range(n_folds):
        train_end = start_train + k * fold_sizes
        val_end = start_train + (k + 1) * fold_sizes if k < n_folds - 1 else n
        if train_end < 100 or (val_end - train_end) < 50:
            continue
        tr = np.arange(0, train_end)
        va = np.arange(train_end, val_end)
        folds.append((tr, va))
    return folds

def train_meta_with_oof_tables(
    train_df,
    weights_6,
    alpha,
    n_folds=5,
    seed=42,
    calibrate_max_cv=3
):
    """
    Robust OOF meta training:
    - builds OOF p_tables on train via walk-forward folds
    - trains meta model (LogReg), optionally calibrated if enough samples per class
    Returns (meta_model, diagnostics dict)
    """

    folds = make_walkforward_folds(train_df.index, n_folds=n_folds, min_train=0.5)
    if len(folds) < 2:
        raise ValueError("Недостатньо даних для walk-forward OOF. Зменшіть n_folds або збільшіть train.")

    # OOF probs [G,D,S]
    p_oof = np.full((len(train_df), 3), np.nan, dtype=float)
    y = train_df["target_3"].astype(int).values

    for tr_pos, va_pos in folds:
        tr = train_df.iloc[tr_pos].copy()
        va = train_df.iloc[va_pos].copy()

        fb = tr["target_3"].value_counts(normalize=True).reindex([2,0,1]).fillna(1/3).values

        fold_tables = build_tables_from_train_df(tr, alpha=alpha)
        p_va = predict_tables_for_df(va, fold_tables, weights_6, fb)  # [G,D,S]
        p_oof[va_pos] = p_va

    mask = ~np.isnan(p_oof).any(axis=1)
    X_meta = probs_to_logits(p_oof[mask])   # shape (n,2)
    y_meta = y[mask]

    # diagnostics for tables OOF
    ll_tables_oof = log_loss(y_meta, gds_to_dsg(p_oof[mask]), labels=[0,1,2])

    # base meta model
    base_lr = LogisticRegression(
        max_iter=4000,
        C=1.0,
        class_weight="balanced",
        random_state=seed
    )

    # --- adaptive calibration ---
    counts = np.bincount(y_meta, minlength=3)
    min_count = int(counts.min())

    if min_count >= 2:
        cv_cal = min(calibrate_max_cv, min_count)  # 2 or 3
        meta = CalibratedClassifierCV(base_lr, method="sigmoid", cv=cv_cal)
        meta.fit(X_meta, y_meta)
        p_meta_oof = meta.predict_proba(X_meta)  # [D,S,G] by labels 0,1,2
        ll_meta_oof = log_loss(y_meta, p_meta_oof, labels=[0,1,2])
        calibrated = True
        cal_cv_used = cv_cal
    else:
        # too few samples in some class -> no calibration
        base_lr.fit(X_meta, y_meta)
        meta = base_lr
        p_meta_oof = base_lr.predict_proba(X_meta)
        ll_meta_oof = log_loss(y_meta, p_meta_oof, labels=[0,1,2])
        calibrated = False
        cal_cv_used = None

    return meta, {
        "ll_tables_oof": float(ll_tables_oof),
        "ll_meta_oof": float(ll_meta_oof),
        "n_oof": int(mask.sum()),
        "meta_calibrated": calibrated,
        "calibration_cv": cal_cv_used,
        "class_counts_oof": counts.tolist()
    }

def predict_meta_from_tables(meta, d_df, tables, weights_6, fallback_gds):
    p_tab_gds = predict_tables_for_df(d_df, tables, weights_6, fallback_gds)
    X = probs_to_logits(p_tab_gds)
    proba_dsg = meta.predict_proba(X)  # [D,S,G]
    pD, pS, pG = proba_dsg[:, 0], proba_dsg[:, 1], proba_dsg[:, 2]
    p_gds = np.vstack([pG, pD, pS]).T
    return safe_clip_probs(p_gds)

# ---------------------------
# Evaluate one ticker
# ---------------------------
def evaluate_one_ticker(
    stock_ticker,
    market_ticker,
    start_date,
    end_date,
    params,
    weights_6,
    train_frac=0.8,
    seed=42
):
    stock, market = download_two_tickers(stock_ticker, market_ticker, start_date, end_date)
    d_all = build_ready_df(
        stock_df=stock,
        market_df=market,
        horizon=params["horizon"],
        lookback_high=params["lookback_high"],
        theta_mode=params["theta_mode"],
        theta_k=params["theta_k"],
    )
    split = int(len(d_all) * train_frac)
    train = d_all.iloc[:split].copy()
    test  = d_all.iloc[split:].copy()

    fb = train["target_3"].value_counts(normalize=True).reindex([2,0,1]).fillna(1/3).values
    tables = build_tables_from_train_df(train, alpha=params["alpha"])

    # tables-only
    y_true = test["target_3"].astype(int).values
    p_tab = predict_tables_for_df(test, tables, weights_6, fb)
    proba_tab_dsg = gds_to_dsg(p_tab)

    ll_tab = log_loss(y_true, proba_tab_dsg, labels=[0,1,2])
    acc_tab = accuracy_score(y_true, np.argmax(proba_tab_dsg, axis=1))
    br_tab = brier_multiclass(y_true, proba_tab_dsg)

    # meta
    meta, diag = train_meta_with_oof_tables(train, weights_6, alpha=params["alpha"], n_folds=5, seed=seed)
    p_meta = predict_meta_from_tables(meta, test, tables, weights_6, fb)
    proba_meta_dsg = gds_to_dsg(p_meta)

    ll_meta = log_loss(y_true, proba_meta_dsg, labels=[0,1,2])
    acc_meta = accuracy_score(y_true, np.argmax(proba_meta_dsg, axis=1))
    br_meta = brier_multiclass(y_true, proba_meta_dsg)

    use_meta = ll_meta < ll_tab

    return {
        "ticker": stock_ticker,
        "n_total": len(d_all),
        "n_train": len(train),
        "n_test": len(test),

        "tables_logloss": ll_tab,
        "tables_accuracy": acc_tab,
        "tables_brier": br_tab,

        "meta_logloss": ll_meta,
        "meta_accuracy": acc_meta,
        "meta_brier": br_meta,

        "delta_logloss": ll_meta - ll_tab,
        "use_meta": use_meta,

        "oof_tables_logloss": diag["ll_tables_oof"],
        "oof_meta_logloss": diag["ll_meta_oof"],
        "oof_n": diag["n_oof"],
    }, {"tables": tables, "meta": meta, "train": train, "test": test, "fallback": fb}

# ---------------------------
# Forecast table + interpretation
# ---------------------------
def prediction_table_with_components(
    stock_ticker,
    market_ticker,
    start_date,
    end_date,
    params,
    weights_6,
    use_meta=False,
    alpha=1.0,
    train_frac=0.8,
    seed=42
):
    # build dataset
    stock, market = download_two_tickers(stock_ticker, market_ticker, start_date, end_date)
    d_all = build_ready_df(stock, market, params["horizon"], params["lookback_high"], params["theta_mode"], params["theta_k"])

    split = int(len(d_all) * train_frac)
    train = d_all.iloc[:split].copy()
    test  = d_all.iloc[split:].copy()

    fb = train["target_3"].value_counts(normalize=True).reindex([2,0,1]).fillna(1/3).values
    tables = build_tables_from_train_df(train, alpha=alpha)

    # per-row component probabilities (interpretation)
    comps = {}
    for name, col in [("dist","dist_bucket"),("rsi","rsi_bucket"),("macd","macd_bucket"),
                      ("vol","vol_bucket"),("trend","trend_bucket"),("market","market_bucket")]:
        tbl = tables[name]
        # map each bucket to row probabilities
        # fallback if missing
        p = []
        for b in test[col].values:
            if b in tbl.index:
                r = tbl.loc[b][["Growth","Decline","Sideways"]].values.astype(float)
            else:
                r = fb
            p.append(r)
        comps[name] = np.vstack(p)  # (n,3) [G,D,S]

    # tables mix
    p_tab = predict_tables_for_df(test, tables, weights_6, fb)  # [G,D,S]

    # meta if needed
    if use_meta:
        meta, _diag = train_meta_with_oof_tables(train, weights_6, alpha=alpha, n_folds=5, seed=seed)
        p_final = predict_meta_from_tables(meta, test, tables, weights_6, fb)
    else:
        p_final = p_tab

    proba_dsg = gds_to_dsg(p_final)  # [D,S,G]
    y_true = test["target_3"].astype(int).values
    y_pred = np.argmax(proba_dsg, axis=1)

    sorted_p = np.sort(proba_dsg, axis=1)
    confidence = sorted_p[:, -1]
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    out = pd.DataFrame({
        "date": test.index,
        "ticker": stock_ticker,
        "P_Decline": proba_dsg[:,0],
        "P_Sideways": proba_dsg[:,1],
        "P_Growth": proba_dsg[:,2],
        "predicted_class": y_pred,
        "true_class": y_true,
        "correct": (y_pred == y_true).astype(int),
        "confidence": confidence,
        "margin": margin,
        "model_used": ("DDL+Meta" if use_meta else "DDL")
    })

    # add “розкриття прогнозів” (компоненти)
    # наприклад: для Growth компонентів
    for name in ["dist","rsi","macd","vol","trend","market"]:
        out[f"{name}_G"] = comps[name][:,0]
        out[f"{name}_D"] = comps[name][:,1]
        out[f"{name}_S"] = comps[name][:,2]

    return out

