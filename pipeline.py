import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

from tune import tune_one_ticker_optuna
from core import (
    evaluate_one_ticker,
    download_two_tickers,
    build_ready_df,
    build_tables_from_train_df,
    train_meta_with_oof_tables,
    predict_tables_for_df
)

from pipeline_io import (
    save_optuna_cache,
    save_bundle,
    load_optuna_cache,
    load_bundle
)

def run_one_ticker_full_pipeline(
    stock_ticker,
    market_ticker,
    start_date,
    end_date,
    cache_dir_optuna="../artifacts/optuna_cache",
    cache_dir_bundle="../artifacts/bundle_cache",
    ttl_days=30,
    train_frac=0.8,
    n_trials=80,
    seed=42
):
    # 0) Якщо є "final bundle" і він свіжий — можна одразу повернути його (для продукту).
    cached_final = load_bundle(cache_dir_bundle, stock_ticker, kind="final", ttl_days=ttl_days)
    if cached_final is not None:
        print(f"✅ Loaded FINAL bundle from cache for {stock_ticker} (created_at={cached_final['meta']['created_at']})")
        return {"final_bundle": cached_final, "research_metrics": None, "best_params": None, "weights_6": None}

    # 1) Optuna cache
    opt_cached = load_optuna_cache(cache_dir_optuna, stock_ticker, ttl_days=ttl_days)
    if opt_cached is not None:
        best_params = opt_cached["best_params"]
        weights_6 = opt_cached["weights_6"]
        best_value = opt_cached["best_value"]
        print(f"✅ Loaded Optuna cache for {stock_ticker}: CV logloss={best_value:.6f}")
    else:
        study, best_params, weights_6 = tune_one_ticker_optuna(
            stock_ticker=stock_ticker,
            market_ticker=market_ticker,
            start_date=start_date,
            end_date=end_date,
            train_frac=train_frac,
            n_trials=n_trials,
            seed=seed
        )
        best_value = float(study.best_value)
        save_optuna_cache(cache_dir_optuna, stock_ticker, best_value, best_params, weights_6)
        print(f"💾 Saved Optuna cache for {stock_ticker}: CV logloss={best_value:.6f}")

    # 2) Research evaluation on test (щоб визначити use_meta)
    metrics, pack = evaluate_one_ticker(
        stock_ticker=stock_ticker,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date,
        params=best_params,
        weights_6=weights_6,
        train_frac=train_frac,
        seed=seed
    )
    print(f"📌 {stock_ticker} test: tables_ll={metrics['tables_logloss']:.6f}, meta_ll={metrics['meta_logloss']:.6f}, use_meta={metrics['use_meta']}")

    # 3) Save RESEARCH bundle (tables on TRAIN only + meta model trained via OOF on TRAIN)
    # pack already has tables/meta that were trained for evaluation (train split only)
    # We save for reproducibility of research.
    save_bundle(
        base_dir=cache_dir_bundle,
        ticker=stock_ticker,
        kind="research",
        tables=pack["tables"],
        meta_model=pack["meta"],
        meta_fields={
            "cv_logloss": best_value,
            "train_frac": train_frac,
            "use_meta": bool(metrics["use_meta"]),
            "tables_logloss_test": float(metrics["tables_logloss"]),
            "meta_logloss_test": float(metrics["meta_logloss"]),
            "tables_accuracy_test": float(metrics["tables_accuracy"]),
            "meta_accuracy_test": float(metrics["meta_accuracy"]),
            "tables_brier_test": float(metrics["tables_brier"]),
            "meta_brier_test": float(metrics["meta_brier"]),
            **{f"p_{k}": v for k, v in best_params.items()},
            **{f"w_{k}": v for k, v in weights_6.items()},
        }
    )
    print(f"💾 Saved RESEARCH bundle for {stock_ticker}")

    # 4) Train FINAL bundle on ALL data (train+test)
    s_df, m_df = download_two_tickers(stock_ticker, market_ticker, start_date, end_date)
    d_all = build_ready_df(
        s_df, m_df,
        horizon=best_params["horizon"],
        lookback_high=best_params["lookback_high"],
        theta_mode=best_params["theta_mode"],
        theta_k=best_params["theta_k"],
    )

    # tables on ALL
    tables_all = build_tables_from_train_df(d_all, alpha=best_params["alpha"])

    # meta on ALL (OOF within all)
    fb_all = d_all["target_3"].value_counts(normalize=True).reindex([2,0,1]).fillna(1/3).values
    meta_all, diag_all = train_meta_with_oof_tables(
        train_df=d_all,
        weights_6=weights_6,
        alpha=best_params["alpha"],
        n_folds=5,
        seed=seed
    )

    # if research said meta not useful, you can choose to not use it in final:
    final_meta_model = meta_all if metrics["use_meta"] else None

    save_bundle(
        base_dir=cache_dir_bundle,
        ticker=stock_ticker,
        kind="final",
        tables=tables_all,
        meta_model=final_meta_model,
        meta_fields={
            "cv_logloss": best_value,
            "use_meta": bool(metrics["use_meta"]),
            "oof_tables_logloss_all": float(diag_all["ll_tables_oof"]),
            "oof_meta_logloss_all": float(diag_all["ll_meta_oof"]),
            "oof_n_all": int(diag_all["n_oof"]),
            **{f"p_{k}": v for k, v in best_params.items()},
            **{f"w_{k}": v for k, v in weights_6.items()},
        }
    )
    print(f"💾 Saved FINAL bundle for {stock_ticker}")

    final_bundle = load_bundle(cache_dir_bundle, stock_ticker, kind="final", ttl_days=99999)
    return {"final_bundle": final_bundle, "research_metrics": metrics, "best_params": best_params, "weights_6": weights_6}


run_one_ticker_full_pipeline(
    stock_ticker="AMD",
    market_ticker="SPY",
    start_date="2010-01-01",
    end_date=None
)