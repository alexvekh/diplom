import numpy as np
import pandas as pd
import yfinance as yf

from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator


# ================= CONFIG =================

STOCK = "AMD"
MARKET = "SPY"

START_DATE = "2010-01-01"
END_DATE = None

HORIZON = 120
LOOKBACK_HIGH = 252

THETA_MODE = "atr"   # "atr" or "vol"
THETA_K = 0.7

ALPHA = 1.0  # smoothing (Laplace)

WEIGHTS = {
    "dist":   0.5,
    "rsi":    2.2,
    "macd":   1.2,
    "vol":    1.0,
    "trend":  1.2,
    "market": 0.8,
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
    # 2=growth, 1=sideways, 0=decline
    return pd.Series(
        np.select(
            [future_ret > theta, future_ret < -theta],
            [2, 0],
            default=1
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


# ================= TABLES =================

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

def aggregate_weighted(comp: dict):
    keys = ["dist", "rsi", "macd", "vol", "trend", "market"]
    mat = np.vstack([comp[k].values for k in keys])
    w = np.array([WEIGHTS[k] for k in keys], dtype=float)
    final = np.average(mat, axis=0, weights=w)
    final = final / final.sum()
    return pd.Series(final, index=["Growth", "Decline", "Sideways"])


# ================= PIPELINE =================

def train_tables_and_last_row():
    s_df, m_df = download_two_tickers(STOCK, MARKET, START_DATE, END_DATE)

    s = create_stock_features(s_df)
    m = create_market_features(m_df)

    s = s.join(m[["spy_ret", "spy_trend_regime", "spy_dd_pct"]], how="left")

    s["future_ret"] = s["close"].shift(-HORIZON) / s["close"] - 1

    if THETA_MODE == "atr":
        s["theta"] = THETA_K * s["atr_pct"]
    else:
        s["theta"] = THETA_K * s["volatility_20"]

    s["target_3"] = make_target_3(s["future_ret"], s["theta"])

    roll_high = s["close"].rolling(LOOKBACK_HIGH).max()
    s["dd_pct"] = (roll_high - s["close"]) / roll_high * 100

    s["dist_bucket"] = bucket_distance_from_high(s["dd_pct"])
    s["rsi_bucket"] = bucket_rsi(s["rsi"])
    s["macd_bucket"] = macd_status(s["close"])
    s["trend_bucket"] = trend_bucket(s["close"], s["ma200"], s["ma50"])
    s["vol_bucket"] = bucket_volume(s["vol_ratio"], s["close"], s["open"])
    s["market_bucket"] = market_bucket(s["spy_trend_regime"], s["spy_dd_pct"])

    s = s.dropna()

    tables = {
        "dist": prob_table_3class(s["dist_bucket"], s["target_3"], alpha=ALPHA),
        "rsi": prob_table_3class(s["rsi_bucket"], s["target_3"], alpha=ALPHA),
        "macd": prob_table_3class(s["macd_bucket"], s["target_3"], alpha=ALPHA),
        "vol": prob_table_3class(s["vol_bucket"], s["target_3"], alpha=ALPHA),
        "trend": prob_table_3class(s["trend_bucket"], s["target_3"], alpha=ALPHA),
        "market": prob_table_3class(s["market_bucket"], s["target_3"], alpha=ALPHA),
    }

    return tables, s.iloc[-1]


def print_component_line(name: str, info: str, p: pd.Series, n: int):
    print(
        f"{name}: {info}  n={n}  "
        f"Probabilities Growth: {p['Growth']*100:.0f}% "
        f"Decline: {p['Decline']*100:.0f}% "
        f"Sideways: {p['Sideways']*100:.0f}%"
    )


if __name__ == "__main__":
    tables, last = train_tables_and_last_row()

    p_dist, n_dist = current_probs(tables["dist"], last["dist_bucket"])
    p_vol,  n_vol  = current_probs(tables["vol"], last["vol_bucket"])
    p_macd, n_macd = current_probs(tables["macd"], last["macd_bucket"])
    p_rsi,  n_rsi  = current_probs(tables["rsi"], last["rsi_bucket"])
    p_trnd, n_trnd = current_probs(tables["trend"], last["trend_bucket"])
    p_mkt,  n_mkt  = current_probs(tables["market"], last["market_bucket"])

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
    print_component_line("Trend", f"{last['trend_bucket']}", p_trnd, n_trnd)
    print_component_line("Market", f"{last['market_bucket']}", p_mkt, n_mkt)

    comp = {
        "dist": p_dist,
        "rsi": p_rsi,
        "macd": p_macd,
        "vol": p_vol,
        "trend": p_trnd,
        "market": p_mkt
    }

    final = aggregate_weighted(comp)

    print("\n📈 Final probabilities (weighted):")
    print(f"Growth:   {final['Growth']*100:.2f}%")
    print(f"Decline:  {final['Decline']*100:.2f}%")
    print(f"Sideways: {final['Sideways']*100:.2f}%")