# app.py
import streamlit as st
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from product import (
    ProductConfig,
    product_predict_ticker,
    ensure_trained_bundle,
    explain_prediction_row,
)

BIG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

st.set_page_config(page_title="Regime Probabilities", layout="wide")
st.title("📊 Stock Regime Probabilities (DDL / DDL+Meta)")

# --- session state for background training
if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=2)
if "train_futures" not in st.session_state:
    st.session_state.train_futures = {}  # ticker -> future


def parse_tickers(text: str):
    parts = [x.strip().upper() for x in text.replace(";", ",").split(",")]
    return [p for p in parts if p]


default_text = ", ".join(BIG7)
tickers_text = st.text_input("Tickers (comma-separated)", default_text)
tickers = parse_tickers(tickers_text)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    auto_train = st.checkbox("Auto-train if missing", value=True)
with colB:
    only_test = st.checkbox("Show test segment only", value=True)
with colC:
    st.caption("Если bundle нет: появится статус Waiting… (обучение стартует один раз). Затем нажми Refresh.")

cfg = ProductConfig()

if st.button("Run / Refresh"):
    st.session_state["run_now"] = True

run_now = st.session_state.get("run_now", False)

if run_now:
    rows = []
    pred_store = {}   # ticker -> pred_df
    bundle_store = {} # ticker -> bundle (for explanation)

    for t in tickers:
        out = product_predict_ticker(t, cfg=cfg, as_of_test_only=only_test)

        if out["status"] == "missing":
            status = "No data. Waiting for train process…"
            # запуск обучения один раз
            if auto_train and t not in st.session_state.train_futures:
                fut = st.session_state.executor.submit(ensure_trained_bundle, t, cfg)
                st.session_state.train_futures[t] = fut
        else:
            # если обучение было запущено и уже готово — можно убрать future
            if t in st.session_state.train_futures and st.session_state.train_futures[t].done():
                st.session_state.train_futures.pop(t, None)

            status = "OK"
            pred_store[t] = out["pred_df"]
            bundle_store[t] = out["bundle"]

        # summary row (берём последний прогноз если есть)
        if out["status"] == "ok" and len(out["pred_df"]) > 0:
            last = out["pred_df"].iloc[-1]
            cls_map = {0: "Decline", 1: "Sideways", 2: "Growth"}
            rows.append({
                "ticker": t,
                "date": pd.to_datetime(last["date"]).date(),
                "pred": cls_map[int(last["predicted_class"])],
                "P_Growth": float(last["P_Growth"]),
                "P_Decline": float(last["P_Decline"]),
                "P_Sideways": float(last["P_Sideways"]),
                "confidence": float(last["confidence"]),
                "margin": float(last["margin"]),
                "model": last["model_used"],
                "status": status,
            })
        else:
            rows.append({
                "ticker": t,
                "date": None,
                "pred": None,
                "P_Growth": np.nan,
                "P_Decline": np.nan,
                "P_Sideways": np.nan,
                "confidence": np.nan,
                "margin": np.nan,
                "model": None,
                "status": status,
            })

    df_summary = pd.DataFrame(rows)

    st.subheader("Results (latest)")
    st.dataframe(
        df_summary.sort_values(["status", "ticker"]),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --- Selection UI (надежный вариант без “клик по строке”)
    # (у Streamlit selection в dataframe зависит от версии, поэтому делаем через selectbox)
    available = [t for t in tickers if t in pred_store and len(pred_store[t]) > 0]
    if len(available) == 0:
        st.info("Пока нет готовых прогнозов. Нажми Refresh позже (или выключи Auto-train).")
    else:
        left, right = st.columns([1, 2])

        with left:
            sel_ticker = st.selectbox("Ticker for details", available, index=0)

            pred_df = pred_store[sel_ticker]
            # выберем дату (строку) — по умолчанию последняя
            dates = pred_df["date"].astype(str).tolist()
            sel_date = st.selectbox("Pick a prediction row (date)", dates, index=len(dates) - 1)

        with right:
            st.subheader(f"Predictions table: {sel_ticker}")
            st.dataframe(pred_df.tail(250), use_container_width=True, hide_index=True)

        # find row
        row = pred_df[pred_df["date"].astype(str) == sel_date].iloc[0]
        bundle = bundle_store[sel_ticker]

        st.subheader("Расшифровка прогноза (компоненты)")
        expl = explain_prediction_row(bundle, row)

        # аккуратно “отделить” от основной таблицы
        st.markdown("##### Детали по компонентам")
        st.dataframe(expl, use_container_width=True, hide_index=True)

        st.caption("Growth/Decline/Sideways — вероятности из таблиц (в процентах) для текущего bucket-состояния.")