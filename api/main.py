# # Feature Engineering (базовий)
df[(STOCK, "SMA_20")] = df[(STOCK, "Close")].rolling(20).mean()
df[(STOCK, "SMA_50")] = df[(STOCK, "Close")].rolling(50).mean()

ret_stock = df[(STOCK, "Close")].pct_change()
ret_spy   = df[(MARKET,  "Close")].pct_change()

df[(STOCK, "return")] = ret_stock
df[(MARKET,  "return")] = ret_spy

# Relative return
df[(STOCK, "rel_return")] = ret_stock - ret_spy

# Price vs SPY
df[(STOCK, "price_vs_spy")] = (
    df[(STOCK, "Close")] / df[(MARKET, "Close")]
)

# 🔹 Beta (60d)
df[(STOCK, "beta_60d")] = (
    ret_stock.rolling(60).cov(ret_spy) /
    ret_spy.rolling(60).var()
)

# 🔹 Momentum
df[(STOCK, "mom_20")] = df[(STOCK, "Close")].pct_change(20)
df[(MARKET,  "mom_20")] = df[(MARKET,  "Close")].pct_change(20)

df[(STOCK, "rel_mom_20")] = (
    df[(STOCK, "mom_20")] - df[(MARKET, "mom_20")]
)


# df["mom_stock_20"] = df["Close"].pct_change(20)
# df["mom_spy_20"]   = df_spy["Close"].pct_change(20)
# df["rel_mom_20"]   = df["mom_stock_20"] - df["mom_spy_20"]

# Relative features (AAPL vs SPY)
df[(STOCK, "rel_return")] = (
    df[(STOCK, "return")] - df[(MARKET, "return")]
)

# Price ratio
df[(STOCK, "price_vs_spy")] = (
    df[(STOCK, "Close")] / df[(MARKET, "Close")]
)

# Rolling beta (60 днів)
window = 60

cov = (
    df[(STOCK, "return")]
    .rolling(window)
    .cov(df[(MARKET, "return")])
)

var = (
    df[(MARKET, "return")]
    .rolling(window)
    .var()
)

df[(STOCK, "beta_60d")] = cov / var

# Relative momentum
for w in [20, 60]:
    df[(STOCK, f"mom_{w}")] = df[(STOCK, "Close")].pct_change(w)
    df[(MARKET, f"mom_{w}")]  = df[(MARKET, "Close")].pct_change(w)
    df[(STOCK, f"rel_mom_{w}")] = (
        df[(STOCK, f"mom_{w}")] - df[(MARKET, f"mom_{w}")]
    )