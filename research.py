# research.py
# One-ticker research runner with plots + interpretation tables.
# Assumes you already have:
#   - tune.py: tune_one_ticker_optuna(...)
#   - core.py: evaluate_one_ticker(...), prediction_table_with_components(...)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from tune import tune_one_ticker_optuna
from core import evaluate_one_ticker, prediction_table_with_components


# ---------------------------
# Helpers: metric explanations
# ---------------------------
def print_metric_explanations():
    print("\n" + "=" * 80)
    print("📚 ОБЪЯСНЕНИЕ МЕТРИК")
    print("-" * 80)
    print("1) LogLoss (кросс-энтропия):")
    print("   - Чем меньше, тем лучше.")
    print("   - Сильно штрафует модель, которая уверенно ошибается.")
    print("   - Главная метрика качества вероятностей.")
    print("\n2) Accuracy:")
    print("   - Доля правильных классов (argmax по вероятностям).")
    print("   - Не учитывает калибровку вероятностей (насколько модель 'уверена').")
    print("\n3) Multiclass Brier score:")
    print("   - Средний квадрат ошибки вероятностей по классам.")
    print("   - Чем меньше, тем лучше. Хорошо отражает калибровку.")
    print("\n4) Confidence и Margin (в prediction table):")
    print("   - confidence = max(P_D, P_S, P_G) — насколько модель уверена в выбранном классе.")
    print("   - margin = top1 - top2 — 'разрыв' между лучшим и вторым классом.")
    print("=" * 80 + "\n")


def class_name(c: int) -> str:
    return {0: "Decline", 1: "Sideways", 2: "Growth"}.get(int(c), str(c))


# ---------------------------
# Plots
# ---------------------------
def plot_probabilities(pred_df: pd.DataFrame, ticker: str):
    d = pred_df.sort_values("date").copy()

    plt.figure(figsize=(12, 4))
    plt.plot(d["date"], d["P_Growth"], label="P(Growth)")
    plt.plot(d["date"], d["P_Decline"], label="P(Decline)")
    plt.plot(d["date"], d["P_Sideways"], label="P(Sideways)")
    plt.ylim(0, 1)
    plt.title(f"{ticker}: Probabilities over time")
    plt.ylabel("Probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confidence_margin(pred_df: pd.DataFrame, ticker: str):
    d = pred_df.sort_values("date").copy()

    plt.figure(figsize=(12, 4))
    plt.plot(d["date"], d["confidence"], label="confidence (max prob)")
    plt.plot(d["date"], d["margin"], label="margin (top1-top2)")
    plt.ylim(0, 1)
    plt.title(f"{ticker}: Confidence and margin over time")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_accuracy(pred_df: pd.DataFrame, ticker: str):
    d = pred_df.sort_values("date").copy()
    d["cum_acc"] = d["correct"].expanding().mean()

    plt.figure(figsize=(12, 4))
    plt.plot(d["date"], d["cum_acc"])
    plt.ylim(0, 1)
    plt.title(f"{ticker}: Cumulative accuracy over time")
    plt.ylabel("Cumulative accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion(pred_df: pd.DataFrame, ticker: str):
    y_true = pred_df["true_class"].astype(int).values
    y_pred = pred_df["predicted_class"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Decline", "Sideways", "Growth"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"{ticker}: Confusion matrix")
    plt.tight_layout()
    plt.show()


def plot_calibration_growth(pred_df: pd.DataFrame, ticker: str, n_bins: int = 10):
    """
    Calibration curve for Growth (one-vs-rest):
      y = 1 if true_class==2 else 0
      p = P_Growth
    """
    d = pred_df.dropna(subset=["P_Growth", "true_class"]).copy()
    y = (d["true_class"].astype(int).values == 2).astype(int)
    p = d["P_Growth"].astype(float).values

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")

    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{ticker}: Calibration (Growth vs Rest)")
    plt.xlabel("Mean predicted P(Growth)")
    plt.ylabel("Observed frequency of Growth")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Interpretation / component breakdown
# ---------------------------
def show_last_signal_breakdown(pred_df: pd.DataFrame, ticker: str, tail: int = 1):
    """
    Uses columns created by prediction_table_with_components:
      dist_G, dist_D, dist_S, ... etc for each component.
    """
    d = pred_df.sort_values("date").copy()
    last_rows = d.tail(tail).copy()

    comp_names = ["dist", "rsi", "macd", "vol", "trend", "market"]
    rows_out = []

    for _, r in last_rows.iterrows():
        dt = r["date"]
        model_used = r.get("model_used", "")
        pred_c = int(r["predicted_class"])
        true_c = int(r["true_class"])
        rows_out.append({
            "date": dt,
            "model_used": model_used,
            "predicted": class_name(pred_c),
            "true": class_name(true_c),
            "P_Growth": r["P_Growth"],
            "P_Decline": r["P_Decline"],
            "P_Sideways": r["P_Sideways"],
            "confidence": r.get("confidence", np.nan),
            "margin": r.get("margin", np.nan),
        })

    headline = pd.DataFrame(rows_out)
    print("\n" + "=" * 80)
    print("🧾 ПОСЛЕДНИЙ(Е) ПРОГНОЗ(Ы):")
    print(headline.to_string(index=False))
    print("=" * 80)

    # Component table for the last row only (most useful)
    r = d.iloc[-1]
    comp_table = []
    for c in comp_names:
        comp_table.append({
            "component": c,
            "P_Growth": float(r.get(f"{c}_G", np.nan)),
            "P_Decline": float(r.get(f"{c}_D", np.nan)),
            "P_Sideways": float(r.get(f"{c}_S", np.nan)),
        })

    df_comp = pd.DataFrame(comp_table).sort_values("P_Growth", ascending=False)
    print("\n🔍 РАСШИФРОВКА КОМПОНЕНТОВ (последняя дата):")
    print(df_comp.to_string(index=False))


def show_top_confident_mistakes(pred_df: pd.DataFrame, ticker: str, top_n: int = 10):
    d = pred_df.copy()
    d = d[d["correct"] == 0].copy()
    if len(d) == 0:
        print("\n✅ Нет ошибок (на test-сегменте) — нечего показывать.")
        return

    d = d.sort_values("confidence", ascending=False).head(top_n).copy()
    d["predicted_name"] = d["predicted_class"].map(class_name)
    d["true_name"] = d["true_class"].map(class_name)

    cols = ["date", "predicted_name", "true_name", "confidence", "margin", "P_Decline", "P_Sideways", "P_Growth", "model_used"]
    print("\n" + "=" * 80)
    print(f"⚠️ ТОП-{top_n} САМЫХ 'УВЕРЕННЫХ' ОШИБОК ({ticker}):")
    print(d[cols].to_string(index=False))
    print("=" * 80)


# ---------------------------
# Main: one ticker research
# ---------------------------
def run_research_one_ticker(
    stock_ticker: str = "NVDA",
    market_ticker: str = "SPY",
    start_date: str = "2010-01-01",
    end_date=None,
    train_frac: float = 0.8,
    n_trials: int = 80,
    seed: int = 42,
    plot: bool = True,
):
    print_metric_explanations()

    print("\n" + "=" * 80)
    print(f"🔧 OPTUNA TUNING: {stock_ticker}")
    print("=" * 80)

    study, best_params, best_weights_6 = tune_one_ticker_optuna(
        stock_ticker=stock_ticker,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date,
        train_frac=train_frac,
        n_trials=n_trials,
        seed=seed
    )

    print(f"\n✅ Best CV LogLoss: {study.best_value:.6f}")
    print("\n🧩 Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\n⚖️ Best weights_6:")
    for k, v in best_weights_6.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 80)
    print(f"🧪 TEST EVAL (Tables vs Meta): {stock_ticker}")
    print("=" * 80)

    metrics, pack = evaluate_one_ticker(
        stock_ticker=stock_ticker,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date,
        params=best_params,
        weights_6=best_weights_6,
        train_frac=train_frac,
        seed=seed
    )

    # Pretty print
    print("\n📊 METRICS:")
    print(f"  Tables LogLoss: {metrics['tables_logloss']:.6f}")
    print(f"  Meta   LogLoss: {metrics['meta_logloss']:.6f}")
    print(f"  Δ LogLoss (meta - tables): {metrics['delta_logloss']:+.6f}")
    print(f"  Tables Accuracy: {metrics['tables_accuracy']:.4f}")
    print(f"  Meta   Accuracy: {metrics['meta_accuracy']:.4f}")
    print(f"  Tables Brier: {metrics['tables_brier']:.6f}")
    print(f"  Meta   Brier: {metrics['meta_brier']:.6f}")
    print(f"  ✅ use_meta (по LogLoss): {metrics['use_meta']}")
    print(f"  OOF tables LogLoss: {metrics['oof_tables_logloss']:.6f}")
    print(f"  OOF meta   LogLoss: {metrics['oof_meta_logloss']:.6f}")
    print(f"  OOF n: {metrics['oof_n']}")

    # Build prediction table with components (interpretation)
    pred_df = prediction_table_with_components(
        stock_ticker=stock_ticker,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date,
        params=best_params,
        weights_6=best_weights_6,
        use_meta=metrics["use_meta"],
        alpha=best_params["alpha"],
        train_frac=train_frac,
        seed=seed
    )

    # add readable class names
    pred_df = pred_df.copy()
    pred_df["predicted_name"] = pred_df["predicted_class"].map(class_name)
    pred_df["true_name"] = pred_df["true_class"].map(class_name)

    # show last breakdown + top confident mistakes
    show_last_signal_breakdown(pred_df, stock_ticker, tail=1)
    show_top_confident_mistakes(pred_df, stock_ticker, top_n=10)

    # plots
    if plot:
        plot_probabilities(pred_df, stock_ticker)
        plot_confidence_margin(pred_df, stock_ticker)
        plot_cumulative_accuracy(pred_df, stock_ticker)
        plot_confusion(pred_df, stock_ticker)
        plot_calibration_growth(pred_df, stock_ticker, n_bins=10)

    return {
        "study": study,
        "best_params": best_params,
        "best_weights_6": best_weights_6,
        "metrics": metrics,
        "pred_df": pred_df,
        "pack": pack,
    }


# Convenience tickers list (optional)
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "KO", "COST", "AVGO", "AMD"]


if __name__ == "__main__":
    # Run one ticker research (change ticker here)
    out = run_research_one_ticker(
        stock_ticker="NVDA",
        market_ticker="SPY",
        start_date="2010-01-01",
        end_date=None,
        train_frac=0.8,
        n_trials=80,
        seed=42,
        plot=True
    )

    # Save quick artifacts (optional)
    out["pred_df"].to_parquet("artifacts/research_pred_nvda.parquet", index=False)
    pd.DataFrame([out["metrics"]]).to_csv("artifacts/research_metrics_nvda.csv", index=False)