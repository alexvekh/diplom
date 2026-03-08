# prob_tables.py
import numpy as np
import pandas as pd
import yfinance as yf

from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator


# ================= CONFIG =================

STOCK = "GOOGL"
MARKET = "SPY"

START_DATE = "2010-01-01"
END_DATE = None

HORIZON = 30
LOOKBACK_HIGH = 252

THETA_MODE = "atr"   # "atr" or "vol"
THETA_K = 0.4

ALPHA = 1.0  # smoothing (Laplace)

WEIGHTS = {
    "dist":   0.12,
    "rsi":    0.12,
    "rsi_regime": 0.10,
    "macd":   0.08,
    "vol":    0.07,
    "trend":  0.07,
    "market": 0.12,
    "mom":    0.1,
    "ma200":  0.1,
    "volpct": 0.12,
    "spy_mom": 0.1,
}


# ================= DATA =================

def download_two_tickers(stock: str, market: str, start: str, end=None):
    raw = yf.download(
        [stock, market],
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )
    s = raw[stock].copy()
    m = raw[market].copy()
    s.columns = s.columns.str.lower()
    m.columns = m.columns.str.lower()
    return s, m


# ================= FEATURES =================

def create_stock_features(df: pd.DataFrame):
    df = df.copy()

    df["ret_1"] = df["close"].pct_change()
    df["volatility_20"] = df["ret_1"].rolling(20).std()

    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi.rsi()

    vol_mean = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_mean
    
    df["mom_10"] = df["close"].pct_change(10)
    df["mom_20"] = df["close"].pct_change(20)

    return df


def create_market_features(m: pd.DataFrame):
    m = m.copy()
    m["spy_ret"] = m["close"].pct_change()
    m["spy_volatility"] = m["spy_ret"].rolling(20).std()
    m["spy_ma200"] = m["close"].rolling(200).mean()
    m["spy_trend_regime"] = (m["close"] > m["spy_ma200"]).astype(int)

    rolling_max = m["close"].rolling(252).max()
    m["spy_dd_pct"] = (rolling_max - m["close"]) / rolling_max * 100
    return m


# ================= TARGET =================
def make_target_3(future_ret: pd.Series, theta: pd.Series):
    """
    Asymmetric 3-class target: 0 = Decline  (any drop)  1 = Sideways (0 .. +theta) 2 = Growth   (> +theta)
    Symmetric 3-class target:  0 = Decline  (< -theta)  1 = Sideways (between -theta and +theta) 2 = Growth   (> +theta)
    """
    return pd.Series(
        np.select(
            # [future_ret > +theta, future_ret < -theta],  # Symmetric 3-class target:
            [future_ret > +(1.8*theta), future_ret < 0],  # Asymmetric 3-class target:
            [2, 0],                    # Growth,  Decline 
            default=1                  # Sideways (0..theta)
        ),
        index=future_ret.index
    ).astype(int)

# ================= BUCKETS =================

def bucket_distance_from_high(dd_pct: pd.Series):
    bins = [-np.inf, 1, 5, 10, 15, 20, 30, 100, np.inf]
    labels = [
        "New high",
        "Near high",
        "1–5% below",
        "5–10% below",
        "10–15% below",
        "15–20% below",
        "20–30% below",
        ">30% below"
    ]
    return pd.cut(dd_pct, bins=bins, labels=labels, include_lowest=True)


def bucket_rsi(rsi: pd.Series):
    bins = [-np.inf, 30, 45, 55, 70, 80, np.inf]
    labels = ["< 30", "30–45", "45–55", "55–70", "70–80", "> 80"]
    return pd.cut(rsi, bins=bins, labels=labels, include_lowest=True)


def bucket_volume(vol_ratio: pd.Series, close: pd.Series, open_: pd.Series):
    out = pd.Series("Average", index=vol_ratio.index, dtype="object")
    low = vol_ratio < 0.7
    high = vol_ratio > 1.3

    out[low] = "Low"
    out[high] = np.where(
        close[high] > open_[high],
        "High (on growth)",
        "High (on decline)"
    )
    return out


def macd_status(close: pd.Series):
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal

    status = pd.Series(index=close.index, dtype="object")
    above0 = macd > 0

    status[(diff > 0) & (above0)] = "MACD up (above 0)"
    status[(diff > 0) & (~above0)] = "MACD up (below 0)"
    status[(diff <= 0) & (above0)] = "MACD down (above 0)"
    status[(diff <= 0) & (~above0)] = "MACD down (below 0)"
    return status


def trend_bucket(close: pd.Series, ma200: pd.Series, ma50: pd.Series):
    slope = ma50.pct_change(20)
    regime = np.where(close > ma200, "above_MA200", "below_MA200")
    strength = np.where(slope > 0, "rising", "falling")
    return pd.Series(regime + "_" + strength, index=close.index)


def market_bucket(spy_trend_regime: pd.Series, spy_dd_pct: pd.Series):
    near = spy_dd_pct <= 10
    out = pd.Series(index=spy_trend_regime.index, dtype="object")
    out[(spy_trend_regime == 1) & near] = "mkt_bull_near"
    out[(spy_trend_regime == 1) & (~near)] = "mkt_bull_dd"
    out[(spy_trend_regime == 0) & near] = "mkt_bear_near"
    out[(spy_trend_regime == 0) & (~near)] = "mkt_bear_dd"
    return out

def bucket_dist_ma200(dist_ma200: pd.Series):
    # dist_ma200 = (close - ma200) / ma200  (тобто +0.10 = +10% над MA200)
    bins = [-np.inf, -0.20, -0.10, -0.03, 0.03, 0.10, 0.20, np.inf]
    labels = [
        "< -20% below MA200",
        "-20..-10% below",
        "-10..-3% below",
        "-3..+3% near",
        "+3..+10% above",
        "+10..+20% above",
        "> +20% above"
    ]
    return pd.cut(dist_ma200, bins=bins, labels=labels, include_lowest=True)


def bucket_vol_percentile(volatility_20: pd.Series, window: int = 252):
    # percentile rank of current vol vs last 252 days
    vol_pct = volatility_20.rolling(window).rank(pct=True)
    bins = [-np.inf, 0.20, 0.50, 0.80, np.inf]
    labels = ["Low vol", "Normal vol", "High vol", "Crisis vol"]
    return pd.cut(vol_pct, bins=bins, labels=labels, include_lowest=True)


def bucket_spy_momentum(spy_close: pd.Series, lookback: int = 60):
    # 60d SPY momentum (можна змінити)
    spy_mom = spy_close.pct_change(lookback)
    bins = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
    labels = ["Strong down", "Down", "Flat", "Up", "Strong up"]
    return pd.cut(spy_mom, bins=bins, labels=labels, include_lowest=True)

# def bucket_momentum(mom):
#     bins = [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
#     labels = [
#         "Strong down",
#         "Down",
#         "Flat",
#         "Up",
#         "Strong up"
#     ]
#     return pd.cut(mom, bins=bins, labels=labels)

def bucket_momentum(mom: pd.Series):
    bins = [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
    labels = ["Strong down", "Down", "Flat", "Up", "Strong up"]
    return pd.cut(mom, bins=bins, labels=labels, include_lowest=True)

# def mom_bucket(ret_1: pd.Series):
#     # 20-day momentum (можна змінити)
#     mom20 = (1 + ret_1).rolling(20).apply(np.prod, raw=True) - 1
#     bins = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
#     labels = ["Strong down", "Down", "Flat", "Up", "Strong up"]
#     return pd.cut(mom20, bins=bins, labels=labels, include_lowest=True)


# ================= TABLES =================

def prob_table_2class(bucket: pd.Series, y2: pd.Series, alpha: float = 1.0):
    d = pd.DataFrame({"bucket": bucket, "y": y2}).dropna()
    ct = pd.crosstab(d["bucket"], d["y"])
    for c in [0, 1]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[[0, 1]]
    ct.columns = ["NotRise", "Rise"]

    n = ct.sum(axis=1)
    probs = (ct + alpha).div(n + 2 * alpha, axis=0)
    return pd.concat([n.rename("n"), probs], axis=1)


def prob_table_3class(bucket: pd.Series, y3: pd.Series, alpha: float = 1.0):
    d = pd.DataFrame({"bucket": bucket, "y": y3}).dropna()
    ct = pd.crosstab(d["bucket"], d["y"])

    for c in [0, 1, 2]:
        if c not in ct.columns:
            ct[c] = 0

    ct = ct[[0, 1, 2]]
    ct.columns = ["Decline", "Sideways", "Growth"]

    n = ct.sum(axis=1)
    probs = (ct + alpha).div(n + 3 * alpha, axis=0)

    return pd.concat([n.rename("n"), probs], axis=1)


def current_probs(tbl: pd.DataFrame, bucket_value):
    """
    tbl: DataFrame с колонками ['n','Decline','Sideways','Growth']
    Возвращает: (p_series, n_int)
    p_series индекс ['Growth','Decline','Sideways']
    """
    if bucket_value in tbl.index:
        row = tbl.loc[bucket_value]
        n = int(row["n"])
        p = row[["Growth", "Decline", "Sideways"]]
        return p, n
    return pd.Series({"Growth": np.nan, "Decline": np.nan, "Sideways": np.nan}), 0

def normalize_weights(w: dict, keys: list[str]) -> dict:
    out = {k: float(w.get(k, 1.0)) for k in keys}  # default 1.0 for missing
    s = sum(out.values())
    return {k: v/s for k, v in out.items()}


def aggregate_weighted(comp: dict):
    keys = ["dist","rsi","rsi_regime","macd","vol","trend","market","mom","ma200","volpct","spy_mom"]
    mat = np.vstack([comp[k].values for k in keys])
    w = np.array([WEIGHTS[k] for k in keys], dtype=float)
    final = np.average(mat, axis=0, weights=w)
    final = final / final.sum()
    return pd.Series(final, index=["Growth", "Decline", "Sideways"])

def aggregate_log_odds(comp: dict, base_rate):
    
    keys = ["dist","rsi","rsi_regime","macd","vol","trend","market","mom","ma200","volpct","spy_mom"]

    scores = np.zeros(3)

    for k in keys:
        p = comp[k].values
        scores += WEIGHTS[k] * np.log(p / base_rate)

    exp = np.exp(scores)
    probs = exp / exp.sum()

    return pd.Series(probs, index=["Growth","Decline","Sideways"])

# ================= PIPELINE =================

def train_tables_and_last_row():
    s_df, m_df = download_two_tickers(STOCK, MARKET, START_DATE, END_DATE)

    s = create_stock_features(s_df)
    m = create_market_features(m_df)

    s = s.join(m[["spy_ret", "spy_trend_regime", "spy_dd_pct"]], how="left")

    s["future_ret"] = s["close"].shift(-HORIZON) / s["close"] - 1

    # if THETA_MODE == "atr":
    #     s["theta"] = THETA_K * s["atr_pct"]
    # else:
    #     s["theta"] = THETA_K * s["volatility_20"]

    scale = np.sqrt(HORIZON)
    
    if THETA_MODE == "atr":
        s["theta"] = THETA_K * s["atr_pct"] * scale
    else:
        s["theta"] = THETA_K * s["volatility_20"] * scale
    # ---------------------------------------------------------  New
    
    # s["theta"] = 0.15134216589745253 * s["volatility_20"] * np.sqrt(150)



    s["target_3"] = make_target_3(s["future_ret"], s["theta"])

    roll_high = s["close"].rolling(LOOKBACK_HIGH).max()
    s["dd_pct"] = (roll_high - s["close"]) / roll_high * 100

    s["dist_bucket"] = bucket_distance_from_high(s["dd_pct"])
    s["rsi_bucket"] = bucket_rsi(s["rsi"])
    s["macd_bucket"] = macd_status(s["close"])
    s["trend_bucket"] = trend_bucket(s["close"], s["ma200"], s["ma50"])
    s["vol_bucket"] = bucket_volume(s["vol_ratio"], s["close"], s["open"])
    s["market_bucket"] = market_bucket(s["spy_trend_regime"], s["spy_dd_pct"])

    s["mom20_close"] = s["close"].pct_change(20)
    s["mom20_prod"]  = (1+s["ret_1"]).rolling(20).apply(np.prod, raw=True) - 1
    s["mom20"] = s["close"].pct_change(20)
    s["mom_bucket"] = bucket_momentum(s["mom20"])

    s["dist_ma200"] = (s["close"] - s["ma200"]) / s["ma200"]
    s["ma200_bucket"] = bucket_dist_ma200(s["dist_ma200"])

    # --- NEW FEATURE 2: volatility percentile regime ---
    s["volpct_bucket"] = bucket_vol_percentile(s["volatility_20"], window=252)

    # --- NEW FEATURE 3: SPY momentum bucket (60d) ---
    # беремо momentum зі SPY close з market df (m)
    m["spy_mom_bucket"] = bucket_spy_momentum(m["close"], lookback=60)
    s = s.join(m[["spy_mom_bucket"]], how="left")
    s["rsi_regime_bucket"] = (s["rsi_bucket"].astype(str) + "_" + s["spy_trend_regime"].astype(str))


    print(s[["mom20_close","mom20_prod"]].dropna().corr())

    s = s.dropna()

    print("\n=== MOMENTUM vs FUTURE RETURN ===")
    print(s.groupby("mom_bucket")["future_ret"].agg(["count","mean","median"]))
    print(s.groupby("mom_bucket", observed=True)["future_ret"].agg(["count","mean","median"]))
    print("\n=== MOMENTUM vs TARGET ===")
    print(pd.crosstab(s["mom_bucket"], s["target_3"], normalize="index"))

    g = s.groupby("mom_bucket")["future_ret"].agg(
    count="count",
    mean="mean",
    median="median"
    )
    g["share_gt0"] = s.groupby("mom_bucket")["future_ret"].apply(lambda x: (x>0).mean())
    print(g)

    tables = {
        "dist": prob_table_3class(s["dist_bucket"], s["target_3"], alpha=ALPHA),
        "rsi": prob_table_3class(s["rsi_bucket"], s["target_3"], alpha=ALPHA),
        "rsi_regime": prob_table_3class(s["rsi_regime_bucket"], s["target_3"], alpha=ALPHA),
        "macd": prob_table_3class(s["macd_bucket"], s["target_3"], alpha=ALPHA),
        "vol": prob_table_3class(s["vol_bucket"], s["target_3"], alpha=ALPHA),
        "trend": prob_table_3class(s["trend_bucket"], s["target_3"], alpha=ALPHA),
        "market": prob_table_3class(s["market_bucket"], s["target_3"], alpha=ALPHA),
        "mom": prob_table_3class(s["mom_bucket"], s["target_3"], alpha=ALPHA),
        "ma200":  prob_table_3class(s["ma200_bucket"], s["target_3"], alpha=ALPHA),
        "volpct": prob_table_3class(s["volpct_bucket"], s["target_3"], alpha=ALPHA),
        "spy_mom": prob_table_3class(s["spy_mom_bucket"], s["target_3"], alpha=ALPHA),
    }
    print("TARGET DISTRIBUTION:")
    print(s["target_3"].value_counts(normalize=True).sort_index())
    print("theta stats:", s["theta"].describe())
    print("future_ret stats:", s["future_ret"].describe())

    print("share future_ret > theta:", (s["future_ret"] > s["theta"]).mean())
    print("share future_ret < 0:", (s["future_ret"] < 0).mean())
    print("share 0 <= future_ret <= theta:", ((s["future_ret"] >= 0) & (s["future_ret"] <= s["theta"])).mean())

    return tables, s.iloc[-1]


def print_component_line(name: str, info: str, p: pd.Series, n: int):
    print(
        f"{name}: {info}  n={n}  "
        f"Probabilities Growth: {p['Growth']*100:.0f}% "
        f"Decline: {p['Decline']*100:.0f}% "
        f"Sideways: {p['Sideways']*100:.0f}%"
    )


def diagnose_bucket_table(tbl: pd.DataFrame, name: str, min_n: int = 50):
    """
    tbl: DataFrame з колонками
         ['n', 'Decline', 'Sideways', 'Growth']
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC FOR: {name}")
    print(f"{'='*80}")

    t = tbl.copy()

    # base rates weighted by n
    total_n = t["n"].sum()
    base_growth = (t["Growth"] * t["n"]).sum() / total_n
    base_decline = (t["Decline"] * t["n"]).sum() / total_n
    base_sideways = (t["Sideways"] * t["n"]).sum() / total_n

    t["Growth_lift"] = t["Growth"] / base_growth
    t["Decline_lift"] = t["Decline"] / base_decline
    t["Sideways_lift"] = t["Sideways"] / base_sideways

    t["Growth_minus_base"] = t["Growth"] - base_growth
    t["Decline_minus_base"] = t["Decline"] - base_decline
    t["Sideways_minus_base"] = t["Sideways"] - base_sideways

    print("\nBase rates:")
    print(f"Growth   = {base_growth:.3f}")
    print(f"Decline  = {base_decline:.3f}")
    print(f"Sideways = {base_sideways:.3f}")

    print("\nBucket table:")
    print(
        t[[
            "n",
            "Growth", "Decline", "Sideways",
            "Growth_lift", "Decline_lift", "Sideways_lift",
            "Growth_minus_base", "Decline_minus_base", "Sideways_minus_base"
        ]].sort_values("Growth", ascending=False)
    )

    print("\nSummary:")
    print(f"Growth prob range   = {t['Growth'].max() - t['Growth'].min():.3f}")
    print(f"Decline prob range  = {t['Decline'].max() - t['Decline'].min():.3f}")
    print(f"Sideways prob range = {t['Sideways'].max() - t['Sideways'].min():.3f}")

    small = t[t["n"] < min_n]
    if len(small) > 0:
        print(f"\nBuckets with n < {min_n}:")
        print(small[["n", "Growth", "Decline", "Sideways"]].sort_values("n"))
    else:
        print(f"\nNo buckets with n < {min_n}.")

def monotonicity_check(tbl: pd.DataFrame, ordered_index: list, col: str = "Growth"):
    """
    ordered_index: правильний порядок bucket-ів
    """
    x = tbl.reindex(ordered_index)[col]
    diffs = x.diff().dropna()

    print(f"\nMonotonicity check for {col}:")
    print(x)
    print("\nStep differences:")
    print(diffs)

    non_decreasing = (diffs >= 0).all()
    non_increasing = (diffs <= 0).all()

    if non_decreasing:
        print("Monotonic increasing")
    elif non_increasing:
        print("Monotonic decreasing")
    else:
        print("Not monotonic")

def factor_strength_score(tbl: pd.DataFrame):
    """
    Проста евристика:
    range Growth + range Decline,
    штраф за малі n
    """
    t = tbl.copy()
    strength = (
        (t["Growth"].max() - t["Growth"].min()) +
        (t["Decline"].max() - t["Decline"].min())
    )

    penalty = (t["n"] < 50).mean() * 0.2
    return strength - penalty

if __name__ == "__main__":
    tables, last = train_tables_and_last_row()
    print(tables["dist"].head())

    p_dist, n_dist = current_probs(tables["dist"], last["dist_bucket"])
    p_vol,  n_vol  = current_probs(tables["vol"], last["vol_bucket"])
    p_macd, n_macd = current_probs(tables["macd"], last["macd_bucket"])
    p_rsi,  n_rsi  = current_probs(tables["rsi"], last["rsi_bucket"])
    p_rsi_regime, n_rsi_regime = current_probs(tables["rsi_regime"], last["rsi_regime_bucket"])
    p_trnd, n_trnd = current_probs(tables["trend"], last["trend_bucket"])
    p_mkt,  n_mkt  = current_probs(tables["market"], last["market_bucket"])
    p_mom, n_mom = current_probs(tables["mom"], last["mom_bucket"])
    p_ma200, n_ma200 = current_probs(tables["ma200"], last["ma200_bucket"])
    p_volpct, n_volpct = current_probs(tables["volpct"], last["volpct_bucket"])
    p_spymom, n_spymom = current_probs(tables["spy_mom"], last["spy_mom_bucket"])

    print(f"\n📊 Ticker: {STOCK}")

    print_component_line(
        "Distance from high",
        f"{last['dd_pct']:.2f}% ({last['dist_bucket']})",
        p_dist,
        n_dist
    )
    print_component_line("Volume", f"{last['vol_bucket']}", p_vol, n_vol)
    print_component_line("MACD", f"{last['macd_bucket']}", p_macd, n_macd)
    print_component_line("RSI", f"{last['rsi']:.1f} ({last['rsi_bucket']})", p_rsi, n_rsi)
    print_component_line("RSI", f"{last['rsi']:.1f} ({last['rsi_bucket']})", p_rsi, n_rsi)
    print_component_line("Trend", f"{last['trend_bucket']}", p_trnd, n_trnd)
    print_component_line("Market", f"{last['market_bucket']}", p_mkt, n_mkt)
    print_component_line("Dist to MA200", f"{last['dist_ma200']*100:.1f}% ({last['ma200_bucket']})", p_ma200, n_ma200)
    print_component_line("Vol percentile", f"{last['volpct_bucket']}", p_volpct, n_volpct)
    print_component_line("SPY momentum", f"{last['spy_mom_bucket']}", p_spymom, n_spymom)

    comp = {
        "dist": p_dist,
        "rsi": p_rsi,
        "rsi_regime": p_rsi_regime,
        "macd": p_macd,
        "vol": p_vol,
        "trend": p_trnd,
        "market": p_mkt,
        "mom": p_mom,
        "ma200": p_ma200,
        "volpct": p_volpct,
        "spy_mom": p_spymom,
    }

    # final = aggregate_weighted(comp)
    # base_rate = np.array([0.57, 0.31, 0.12])
    # base_rate = s["target_3"].value_counts(normalize=True).reindex([2,0,1]).values
    base_rate = [1/3, 1/3, 1/3]
    final = aggregate_log_odds(comp, base_rate)

    print("\n📈 Final probabilities (weighted):")
    print(f"Growth:   {final['Growth']*100:.2f}%")
    print(f"Decline:  {final['Decline']*100:.2f}%")
    print(f"Sideways: {final['Sideways']*100:.2f}%")

    for name, tbl in tables.items():
        diagnose_bucket_table(tbl, name=name, min_n=50)

    dist_order = [
        "New high",
        "Near high",
        "1–5% below",
        "5–10% below",
        "10–15% below",
        "15–20% below",
        "20–30% below",
        ">30% below"
    ]

    monotonicity_check(tables["dist"], dist_order, col="Growth")
    monotonicity_check(tables["dist"], dist_order, col="Decline")

    scores = {}
    for name, tbl in tables.items():
        scores[name] = factor_strength_score(tbl)

    print("\nFACTOR STRENGTH RANKING:")
    print(pd.Series(scores).sort_values(ascending=False))