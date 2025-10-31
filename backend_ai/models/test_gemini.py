"""
Energy Forecasting Trainer — SWITCHED TO LEVEL TARGET (kw),
frequency-aware horizon, persistence gating, clean CLI.

MIT License — please keep this header.
"""

from __future__ import annotations
import argparse, json, logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb
import lightgbm as lgb


# --------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("trainer")


# --------------------------- Config ----------------------------------
@dataclass
class TrainConfig:
    train_path: str
    test_path: str
    outdir: str = "artifacts/models"
    horizon_min: int = 15
    valid_size: float = 0.2

    # XGB
    xgb_estimators: int = 1200
    xgb_lr: float = 0.06
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.7
    xgb_early_stopping: int = 150

    # LGB - Best performing tuning
    lgb_objective: str = "regression_l1"
    lgb_estimators: int = 3000
    lgb_lr: float = 0.02
    lgb_num_leaves: int = 40       # Tuned up from 31
    lgb_min_data_in_leaf: int = 8  # Tuned down from 20
    lgb_subsample: float = 0.8
    lgb_colsample: float = 0.8
    lgb_early_stopping: int = 150
    lgb_lambda_l1: float = 0.1     # Tuned down from 1.0
    lgb_lambda_l2: float = 0.1     # Tuned down from 1.0

    random_state: int = 42


# ---------------------- Utility functions ----------------------------
def _normalize_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex named 'ts'.")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df.sort_index()


def _infer_rows_per_min(idx: pd.DatetimeIndex) -> float:
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 1.0
    sp_row = np.median(deltas)
    return 60.0 / max(sp_row, 1)  # rows/min


def _rows_for_minutes(idx: pd.DatetimeIndex, minutes: int) -> int:
    return max(1, int(round(_infer_rows_per_min(idx) * minutes)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


# ---------------------- Feature Engineering --------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    idx = df.index

    # keep current value (for level reconstruction and baseline)
    df["power_kw_cur"] = df["power_kw"]

    # adaptive lags/windows by real frequency
    lag_minutes = [1, 5, 15, 30, 60, 240, 1440]
    lag_rows = [min(_rows_for_minutes(idx, m), n - 1) for m in lag_minutes]
    lag_rows = sorted(set([L for L in lag_rows if L >= 1]))

    for L in lag_rows:
        df[f"lag_{L}"] = df["power_kw"].shift(L)
        # Delta lags retained as they help predict level changes
        df[f"lag_delta_{L}"] = df["power_kw"].diff(L)

    win_minutes = [15, 30, 60, 240, 1440]
    win_rows = [min(_rows_for_minutes(idx, m), n) for m in win_minutes]
    win_rows = sorted(set([w for w in win_rows if w >= 3]))
    for w in win_rows:
        roll = df["power_kw"].rolling(window=w, min_periods=max(3, w // 3))
        df[f"roll_{w}_mean"] = roll.mean()
        df[f"roll_{w}_std"] = roll.std()
        df[f"roll_{w}_min"] = roll.min()
        df[f"roll_{w}_max"] = roll.max()
        df[f"roll_{w}_range"] = df[f"roll_{w}_max"] - df[f"roll_{w}_min"]
        df[f"roll_{w}_cv"] = df[f"roll_{w}_std"] / (df[f"roll_{w}_mean"] + 1e-6)

    # time features (cyclical features retained for seasonality)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["is_weekend"] = idx.dayofweek.isin([5, 6]).astype(int)
    df["is_peak_hour"] = ((idx.hour >= 8) & (idx.hour <= 20)).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * idx.quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * idx.quarter / 4)

    return df


def align_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]
    # drop fully-NaN columns
    bad = [c for c in X.columns if X[c].isna().all()]
    if bad:
        X = X.drop(columns=bad)
    # drop rows with too many NaNs
    mask = y.notna() & (X.isna().mean(axis=1) <= 0.10)
    X, y = X.loc[mask], y.loc[mask]
    return X, y


# ---------------------- Data preparation (Level Target) -----------------------------
def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon_min: int
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int, int]:
    Xtr = create_features(train_df)
    Xte = create_features(test_df)
    common = sorted(set(Xtr.columns) & set(Xte.columns))
    Xtr = Xtr[common]
    Xte = Xte[common]

    # 💡 TARGET IS NOW POWER LEVEL: y(t+H) 💡
    htr = _rows_for_minutes(train_df.index, horizon_min)
    hte = _rows_for_minutes(test_df.index, horizon_min)
    ytr = train_df["power_kw"].shift(-htr) # Target is the future power level
    yte = test_df["power_kw"].shift(-hte) # Target is the future power level

    Xtr, ytr = align_xy(Xtr, ytr)
    Xte, yte = align_xy(Xte, yte)
    return Xtr, Xte, ytr.rename("kw_level"), yte.rename("kw_level"), htr, hte


# ---------------------- Models (No Change) ---------------------------------------
def train_xgb(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> xgb.XGBRegressor:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.valid_size, shuffle=False, random_state=cfg.random_state
    )
    model = xgb.XGBRegressor(
        n_estimators=cfg.xgb_estimators,
        learning_rate=cfg.xgb_lr,
        max_depth=cfg.xgb_max_depth,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,
        objective="reg:absoluteerror",
        early_stopping_rounds=cfg.xgb_early_stopping,
        eval_metric="mae",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> lgb.LGBMRegressor:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.valid_size, shuffle=False, random_state=cfg.random_state
    )
    model = lgb.LGBMRegressor(
        n_estimators=cfg.lgb_estimators,
        learning_rate=cfg.lgb_lr,
        num_leaves=cfg.lgb_num_leaves,
        min_child_samples=cfg.lgb_min_data_in_leaf,
        subsample=cfg.lgb_subsample,
        colsample_bytree=cfg.lgb_colsample,
        objective=cfg.lgb_objective,
        lambda_l1=cfg.lgb_lambda_l1,
        lambda_l2=cfg.lgb_lambda_l2,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=cfg.lgb_early_stopping, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    return model


# ---------------------- Baseline & Evaluation ------------------------
def persistence_baseline(test_df: pd.DataFrame, y_true_level: pd.Series, horizon_rows: int
                         ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Calculates persistence metrics for a LEVEL target."""
    # true future level is already in y_true_level (aligned in prepare_data)
    y_true_level = y_true_level.dropna()
    # predict ŷ = y(t)
    y_pred_persist = test_df["power_kw"].loc[y_true_level.index]
    mets = _metrics(y_true_level.values, y_pred_persist.values)
    return mets, y_true_level.values, y_pred_persist.values


def evaluate_heads(
    heads: Dict[str, np.ndarray],
    y_true_level: np.ndarray, # y_true is the level now
    persist_rmse: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluates heads against the true level."""
    results: Dict[str, Dict[str, float]] = {}
    for name, pred_level in heads.items():
        m = _metrics(y_true_level, pred_level)
        m["skill_vs_persist"] = 1.0 - (m["rmse"] / (persist_rmse + 1e-12))
        results[name] = m
    return results


# ---------------------- Plotting (No Change) -------------------------------------
def plot_predictions(idx: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, title: str, out: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(idx, y_true, label="Actual", lw=1.5)
    ax1.plot(idx, y_pred, "--", label="Predicted", lw=1.2)
    err = y_true - y_pred
    roll = pd.Series(err, index=idx).rolling(96, min_periods=1).std()
    ax1.fill_between(idx, y_pred - 2 * roll.values, y_pred + 2 * roll.values, alpha=0.2, label="≈95% band")
    ax1.legend(); ax1.set_title(title); ax1.set_ylabel("Power (kW)"); ax1.grid(alpha=0.4)
    ax2.scatter(idx, err, s=10, alpha=0.5)
    ax2.axhline(0, color="r", ls="--")
    ax2.set_title("Residuals"); ax2.set_xlabel("Time"); ax2.set_ylabel("Actual - Predicted"); ax2.grid(alpha=0.4)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()


# ---------------------- Main training routine ------------------------
def run(cfg: TrainConfig):
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    # Load
    tr = pd.read_csv(cfg.train_path, parse_dates=["ts"], index_col="ts")
    te = pd.read_csv(cfg.test_path,  parse_dates=["ts"], index_col="ts")
    if "power_kw" not in tr.columns or "power_kw" not in te.columns:
        raise ValueError("Both train and test must contain a 'power_kw' column.")
    tr, te = _normalize_dtindex(tr), _normalize_dtindex(te)

    # Prepare (now returns kw_level target)
    Xtr, Xte, ytr, yte, htr, hte = prepare_data(tr, te, cfg.horizon_min)
    if len(Xtr) < 50 or len(Xte) < 30:
        log.warning(f"Small datasets after alignment: train={len(Xtr)}, test={len(Xte)}")

    # Scale on train only
    scaler = StandardScaler()
    Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr.values), index=Xtr.index, columns=Xtr.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte.values), index=Xte.index, columns=Xte.columns)

    # Train trees on Level
    log.info("Training XGBoost...")
    xgbm = train_xgb(Xtr_s, ytr, cfg)
    log.info("Training LightGBM...")
    lgbm = train_lgb(Xtr_s, ytr, cfg)

    # Test-time Level predictions
    pred_xgb_level = xgbm.predict(Xte_s) # 💡 Prediction is final level
    pred_lgb_level = lgbm.predict(Xte_s) # 💡 Prediction is final level

    # Median and stacked blends
    pred_median_level = np.median(np.vstack([pred_xgb_level, pred_lgb_level]), axis=0)

    # Stacking on Level targets
    stack = Ridge(alpha=0.5, positive=True, random_state=cfg.random_state) # Final alpha=0.5
    
    # Fit stacker on a forward validation split (Level domain)
    Xtr_s_split, Xval_s_split, ytr_split, yval_split = train_test_split(
        Xtr_s, ytr, test_size=cfg.valid_size, shuffle=False, random_state=cfg.random_state
    )
    # 💡 Stack target yval_split is now Level
    stack.fit(
        np.c_[xgbm.predict(Xval_s_split), lgbm.predict(Xval_s_split)],
        yval_split
    )
    # 💡 Stack prediction inputs are Level predictions
    pred_stack_level = stack.predict(np.c_[pred_xgb_level, pred_lgb_level])

    # Persistence baseline & Skill
    # yte now contains the true future power level
    base_mets, y_true_level_aligned, y_persist = persistence_baseline(te.loc[Xte.index], yte, hte)
    log.info(f"PERSISTENCE — MAE {base_mets['mae']:.3f} | RMSE {base_mets['rmse']:.3f} | R² {base_mets['r2']:.3f} | MAPE {base_mets['mape']:.2f}%")

    # Align heads to y_true_level_aligned
    keep = len(y_true_level_aligned)
    heads = {
        "xgb": pred_xgb_level[-keep:],
        "lgb": pred_lgb_level[-keep:],
        "median": pred_median_level[-keep:],
        "stacked": pred_stack_level[-keep:],
        "persistence": y_persist,
    }

    # Evaluate using the true level
    results = evaluate_heads(heads, y_true_level_aligned, persist_rmse=base_mets["rmse"])
    for k, m in results.items():
        log.info(f"{k.upper()} — MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | R² {m['r2']:.3f} | "
                 f"MAPE {m['mape']:.2f}% | Skill {m['skill_vs_persist']:.3%}")

    # Choose production
    best = max(["stacked", "median", "lgb", "xgb"], key=lambda k: results[k]["skill_vs_persist"])
    if results[best]["skill_vs_persist"] <= 0.0:
        production_name = "persistence"
        production_pred = heads["persistence"]
        log.warning(f"No model beat persistence (best={best}, skill={results[best]['skill_vs_persist']:.3%}). Shipping persistence.")
    else:
        production_name = best
        production_pred = heads[best]
        log.info(f"Production head: {production_name} (skill={results[best]['skill_vs_persist']:.3%})")

    # Save artifacts
    joblib.dump(scaler, outdir / f"scaler_{stamp}.pkl")
    joblib.dump(xgbm,  outdir / f"xgb_{stamp}.pkl")
    joblib.dump(lgbm,  outdir / f"lgb_{stamp}.pkl")
    if production_name == "stacked":
        joblib.dump(stack, outdir / f"stacker_{stamp}.pkl")

    # Save metrics & which head shipped
    with open(outdir / f"metrics_{stamp}.json", "w") as f:
        json.dump({"persistence": base_mets, **results, "production": production_name}, f, indent=2)
    (outdir / "production_head.txt").write_text(f"{production_name}\n")

    # Save predictions CSV aligned to evaluation index
    eval_index = Xte.iloc[-keep:].index
    dfp = pd.DataFrame({
        "ts": eval_index,
        "y_true": y_true_level_aligned,
        "y_persistence": heads["persistence"],
        "y_xgb": heads["xgb"],
        "y_lgb": heads["lgb"],
        "y_median": heads["median"],
        "y_stacked": heads["stacked"],
        "y_production": production_pred,
    }).set_index("ts")
    dfp.to_csv(outdir / f"predictions_{production_name}_{stamp}.csv")

    # Plots
    plot_predictions(eval_index, y_true_level_aligned, production_pred,
                     f"Production — {production_name}", outdir / f"plot_production_{stamp}.png")
    plot_predictions(eval_index, y_true_level_aligned, heads["lgb"],
                     "LightGBM", outdir / f"plot_lgb_{stamp}.png")
    plot_predictions(eval_index, y_true_level_aligned, heads["xgb"],
                     "XGBoost", outdir / f"plot_xgb_{stamp}.png")

    log.info("Done.")


# ---------------------- CLI (No Change) ------------------------------------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Energy forecasting trainer (Level Target).")
    p.add_argument("--train", required=True, help="Path to train CSV with columns: ts,power_kw,...")
    p.add_argument("--test",  required=True, help="Path to test CSV with columns: ts,power_kw,...")
    p.add_argument("--outdir", default="artifacts/models", help="Output directory for artifacts.")
    p.add_argument("--horizon-min", type=int, default=15, help="Forecast horizon in minutes.")
    args = p.parse_args()
    return TrainConfig(
        train_path=args.train,
        test_path=args.test,
        outdir=args.outdir,
        horizon_min=args.horizon_min,
    )


if __name__ == "__main__":
    cfg = parse_args()
    log.info(f"Config: {asdict(cfg)}")
    run(cfg)