import pandas as pd
import numpy as np
import argparse, json, logging
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib 

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 

# Attempt to import tree-based models, otherwise, set them to None
try:
    import xgboost as xgb
    import lightgbm as lgb
    TREE_MODELS_AVAILABLE = True
except ImportError:
    xgb, lgb = None, None
    TREE_MODELS_AVAILABLE = False


# --------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("trainer")


# --------------------------- Config ----------------------------------
@dataclass
class TrainConfig:
    FULL_DATA_PATH: str = "smart_manufacturing_data.csv"
    outdir: str = "artifacts/models"
    # FIXED: Reduced horizon_min for better sample retention
    horizon_min: int = 5 
    # FIXED: Reduced valid_size for better sample retention during internal splits
    valid_size: float = 0.1 
    TARGET_COL: str = "energy_consumption"

    # --- ADVANCED XGB CONFIG ---
    xgb_estimators: int = 3000
    xgb_lr: float = 0.03
    xgb_max_depth: int = 5
    xgb_subsample: float = 0.75
    xgb_colsample: float = 0.6
    xgb_lambda_l1: float = 0.5 
    xgb_lambda_l2: float = 0.5 
    xgb_early_stopping: int = 200 

    # --- ADVANCED LGB CONFIG ---
    lgb_objective: str = "regression_l1"
    lgb_estimators: int = 6000
    lgb_lr: float = 0.015
    lgb_num_leaves: int = 40
    lgb_min_data_in_leaf: int = 8 
    lgb_subsample: float = 0.75
    lgb_colsample: float = 0.75
    lgb_early_stopping: int = 200 
    lgb_lambda_l1: float = 0.2 
    lgb_lambda_l2: float = 0.2 
    
    ridge_alpha: float = 1.0 
    random_state: int = 42
    
    # --- STACKER CONFIG ---
    stacker_model: str = "Lasso" 
    stacker_alpha: float = 0.1
    
    # FIXED: Disabled log transformation for stability after sample loss error
    target_transformation: bool = False 


# ---------------------- Utility functions ----------------------------
def _normalize_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp', drop=True)
                df.index = pd.to_datetime(df.index)
            else:
                raise ValueError("Index must be a DatetimeIndex or a column named 'timestamp' must exist.")
            
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    
    df.index.name = None
    
    return df.sort_index()


def _infer_rows_per_min(idx: pd.DatetimeIndex) -> float:
    deltas = idx.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty: return 1.0
    sp_row = np.median(deltas)
    return 60.0 / max(sp_row, 1)


def _rows_for_minutes(idx: pd.DatetimeIndex, minutes: int) -> int:
    return max(1, int(round(_infer_rows_per_min(idx) * minutes)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def _calculate_skill(model_rmse: float, persist_rmse: float) -> float:
    return 1.0 - (model_rmse / (persist_rmse + 1e-12))


# ---------------------- Data Preprocessing --------------------------
def preprocess_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    for col in numerical_cols + [target_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].interpolate(method='time').ffill().bfill()

    cat_cols_to_encode = ['machine_id'] 
    
    for col in df.columns:
        if df[col].dtype == object and col not in cat_cols_to_encode:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except ValueError:
                log.warning(f"Dropping non-numeric string column '{col}' (cannot be converted to float).")
                df = df.drop(columns=[col])

    present_cat_cols = [col for col in cat_cols_to_encode if col in df.columns]
    if present_cat_cols:
         df = pd.get_dummies(df, columns=present_cat_cols, drop_first=True, dtype=int)
    
    cols_to_drop = ['machine_status', 'anomaly_flag', 'predicted_remaining_life', 'failure_type', 'downtime_risk', 'maintenance_required']
    cols_to_drop_present = [col for col in cols_to_drop if col in df.columns]
    
    return df.drop(columns=cols_to_drop_present, errors='ignore')


# ---------------------- Feature Engineering (Enhanced) --------------------------
def create_features(df: pd.DataFrame, target_col: str, horizon_rows: int) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    idx = df.index
    
    df["target_cur"] = df[target_col] 
    
    exog_lag_roll_cols = ['temperature', 'vibration', 'pressure', 'humidity']
    exog_lag_roll_cols = [col for col in exog_lag_roll_cols if col in df.columns and col != target_col] 
    
    # FIXED: Removed ultra-long lags (24*60, 48*60) for data retention/stability
    lag_minutes = [1, 5, 15, 30, 60] 
    lag_rows = [min(_rows_for_minutes(idx, m), n - 1) for m in lag_minutes]
    lag_rows = sorted(set([L for L in lag_rows if L >= 1]))

    for L in lag_rows:
        df[f"target_lag_{L}"] = df[target_col].shift(L)
        df[f"target_lag_delta_{L}"] = df[target_col].diff(L)
        
        # Add a lag close to the forecast horizon H
        if L == 1 and horizon_rows > 1:
            L_H = max(1, horizon_rows - 1) 
            df[f"target_lag_H_{L_H}"] = df[target_col].shift(L_H)
            df[f"target_lag_delta_H_{L_H}"] = df[target_col].diff(L_H)
        
        for col in exog_lag_roll_cols:
            df[f"{col}_lag_{L}"] = df[col].shift(L)
            df[f"{col}_delta_{L}"] = df[col].diff(L)


    win_minutes = [15, 30, 60] 
    win_rows = [min(_rows_for_minutes(idx, m), n) for m in win_minutes] 
    win_rows = sorted(set([w for w in win_rows if w >= 3]))
    
    for w in win_rows:
        roll = df[target_col].rolling(window=w, min_periods=max(3, w // 3))
        # Use causal rolling features (.shift(1))
        df[f"roll_{w}_mean"] = roll.mean().shift(1) 
        df[f"roll_{w}_std"] = roll.std().shift(1)
        
        for col in exog_lag_roll_cols:
            roll_exog = df[col].rolling(window=w, min_periods=max(3, w // 3))
            df[f"{col}_roll_{w}_std"] = roll_exog.std().shift(1) 

    # 3. Time Features (Cyclical)
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["is_weekend"] = idx.dayofweek.isin([5, 6]).astype(int)
    
    df["dom_sin"] = np.sin(2 * np.pi * idx.day / idx.daysinmonth)
    df["dom_cos"] = np.cos(2 * np.pi * idx.day / idx.daysinmonth)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    
    # 4. Drop source columns
    cols_to_drop = [target_col] + exog_lag_roll_cols
    return df.drop(columns=cols_to_drop, errors='ignore')


def align_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    X = X.dropna(axis=1, how='all')
    
    # Alignment: ensures no NaN in features (X) and aligns X and y indices
    idx = X.index.intersection(y.index).intersection(X.dropna(axis=0, how='any').index)
    
    X = X.loc[idx].dropna(axis=1, how='any')
    y = y.loc[X.index]
    
    return X, y


# ---------------------- Data preparation -----------------------------
def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: TrainConfig
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    
    tr_pp = preprocess_data(train_df, cfg.TARGET_COL)
    te_pp = preprocess_data(test_df, cfg.TARGET_COL)
    
    train_cols = set(tr_pp.columns)
    test_cols = set(te_pp.columns)
    all_cols = list(train_cols.union(test_cols))
    
    tr_pp = tr_pp.reindex(columns=all_cols).fillna(0)
    te_pp = te_pp.reindex(columns=all_cols).fillna(0)

    htr = _rows_for_minutes(tr_pp.index, cfg.horizon_min)
    log.info(f"Targeting horizon of {cfg.horizon_min} minutes, which is {htr} rows.")

    Xtr = create_features(tr_pp, cfg.TARGET_COL, htr)
    Xte = create_features(te_pp, cfg.TARGET_COL, htr)

    ytr = train_df[cfg.TARGET_COL].shift(-htr) 
    yte = test_df[cfg.TARGET_COL].shift(-htr) 

    if cfg.target_transformation:
        log.info("Applying log transformation to the target.")
        ytr = np.log1p(ytr)
        yte = np.log1p(yte)

    # --- 🎯 FIXED: Let align_xy handle all NaN removal and alignment ---
    # align_xy properly aligns X and y by finding intersection of valid indices
    Xtr, ytr = align_xy(Xtr, ytr)
    Xte, yte = align_xy(Xte, yte)
    
    # --- CRITICAL: Ensure target variable has no NaN/Inf values after alignment ---
    # XGBoost and other ML models cannot handle NaN/Inf in target variable
    if ytr.isna().any() or np.isinf(ytr).any():
        log.error(f"Target variable ytr still contains NaN/Inf values after alignment. NaN count: {ytr.isna().sum()}, Inf count: {np.isinf(ytr).sum()}")
        # Remove any remaining NaN/Inf values
        valid_idx = ~(ytr.isna() | np.isinf(ytr))
        ytr = ytr[valid_idx]
        Xtr = Xtr.loc[ytr.index]
        
    if yte.isna().any() or np.isinf(yte).any():
        log.error(f"Target variable yte still contains NaN/Inf values after alignment. NaN count: {yte.isna().sum()}, Inf count: {np.isinf(yte).sum()}")
        # Remove any remaining NaN/Inf values
        valid_idx = ~(yte.isna() | np.isinf(yte))
        yte = yte[valid_idx]
        Xte = Xte.loc[yte.index]
    # ------------------------------------------------------------------------------
    
    Xte = Xte.drop(columns=[c for c in Xte.columns if c not in Xtr.columns], errors='ignore')
    Xtr = Xtr.loc[:, Xtr.columns.intersection(Xte.columns)]
    Xte = Xte.loc[:, Xte.columns.intersection(Xtr.columns)]
    
    return Xtr, Xte, ytr.rename("target_level"), yte.rename("target_level"), htr


# ---------------------- Models & Evaluation ----------------------------
def train_xgb(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> xgb.XGBRegressor:
    if not TREE_MODELS_AVAILABLE or xgb is None: return None
    log.info("Training XGBoost (High Capacity)...")
    # train_test_split is now safe with smaller valid_size and more samples
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=cfg.valid_size, shuffle=False, random_state=cfg.random_state)
    model = xgb.XGBRegressor(
        n_estimators=cfg.xgb_estimators, learning_rate=cfg.xgb_lr, max_depth=cfg.xgb_max_depth,
        subsample=cfg.xgb_subsample, colsample_bytree=cfg.xgb_colsample,
        reg_alpha=cfg.xgb_lambda_l1, reg_lambda=cfg.xgb_lambda_l2,
        objective="reg:absoluteerror", early_stopping_rounds=cfg.xgb_early_stopping,
        eval_metric="mae", random_state=cfg.random_state, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> lgb.LGBMRegressor:
    if not TREE_MODELS_AVAILABLE or lgb is None: return None
    log.info("Training LightGBM (High Capacity)...")
    # train_test_split is now safe with smaller valid_size and more samples
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=cfg.valid_size, shuffle=False, random_state=cfg.random_state)
    model = lgb.LGBMRegressor(
        n_estimators=cfg.lgb_estimators, learning_rate=cfg.lgb_lr, num_leaves=cfg.lgb_num_leaves,
        min_child_samples=cfg.lgb_min_data_in_leaf, subsample=cfg.lgb_subsample, colsample_bytree=cfg.lgb_colsample,
        objective=cfg.lgb_objective, lambda_l1=cfg.lgb_lambda_l1, lambda_l2=cfg.lgb_lambda_l2,
        random_state=cfg.random_state, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=cfg.lgb_early_stopping, verbose=False), lgb.log_evaluation(1000)])
    return model

def train_ridge_pipeline(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> Pipeline:
    log.info(f"Training Ridge Regressor (alpha={cfg.ridge_alpha})...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=cfg.ridge_alpha, random_state=cfg.random_state))
    ])
    model.fit(X, y)
    return model


def persistence_baseline(test_df: pd.DataFrame, y_true_level: pd.Series, target_col: str, cfg: TrainConfig
                             ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_true_level = y_true_level.dropna()
    y_pred_persist = test_df[target_col].loc[y_true_level.index]
    
    y_true_level_val = y_true_level.values
    y_pred_persist_val = y_pred_persist.values
    
    if cfg.target_transformation:
        y_true_level_val = np.expm1(y_true_level_val)
        
    mets = _metrics(y_true_level_val, y_pred_persist_val)
    return mets, y_true_level_val, y_pred_persist_val


def inverse_transform_predictions(pred_results: Dict[str, np.ndarray], cfg: TrainConfig) -> Dict[str, np.ndarray]:
    """Inverse transforms tree/linear model predictions if log transformation was used."""
    if cfg.target_transformation:
        for name, pred in pred_results.items():
            if name != 'persistence': 
                pred_results[name] = np.expm1(pred)
                pred_results[name][pred_results[name] < 0] = 0 
    return pred_results


def evaluate_heads(
    heads: Dict[str, np.ndarray],
    y_true_level_aligned: np.ndarray,
    persist_rmse: float,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name, pred_level in heads.items():
        m = _metrics(y_true_level_aligned, pred_level)
        m["skill_vs_persist"] = _calculate_skill(m["rmse"], persist_rmse)
        results[name] = m
    return results

# FIXED: Function definition re-inserted to resolve NameError
def plot_predictions(idx: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, title: str, out: Path):
    target_col = getattr(plot_predictions, 'TARGET_COL', 'Target') 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(idx, y_true, label="Actual", lw=1.5)
    ax1.plot(idx, y_pred, "--", label="Predicted", lw=1.2)
    err = y_true - y_pred
    roll = pd.Series(err, index=idx).rolling(96, min_periods=1).std()
    ax1.fill_between(idx, y_pred - 2 * roll.values, y_pred + 2 * roll.values, alpha=0.2, label="≈95% band")
    ax1.legend(); ax1.set_title(title); ax1.set_ylabel(f"{target_col} (Unit)"); ax1.grid(alpha=0.4)
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
    
    # This line is now safe
    plot_predictions.TARGET_COL = cfg.TARGET_COL 

    # 1. Load & Split
    try:
        df_full = pd.read_csv(cfg.FULL_DATA_PATH) 
        df_full = _normalize_dtindex(df_full)
        log.info(f"Loaded full data from {cfg.FULL_DATA_PATH}. Size: {len(df_full)}")
    except FileNotFoundError:
        log.error(f"FATAL: Dataset '{cfg.FULL_DATA_PATH}' not found. Cannot proceed.")
        return
    except Exception as e:
        log.error(f"FATAL: Error loading/normalizing data: {e}")
        return

    split_point = int(len(df_full) * 0.8)
    tr = df_full.iloc[:split_point].copy() 
    te = df_full.iloc[split_point:].copy()
    log.info(f"Splitting data: Train size {len(tr)}, Test size {len(te)}.")

    # 2. Prepare Data
    Xtr, Xte, ytr, yte, htr = prepare_data(tr, te, cfg)

    if len(Xtr) == 0 or len(Xte) == 0:
        log.error(f"FATAL: Data preparation resulted in empty sets. Cannot proceed.")
        return
        
    log.info(f"Data prepared successfully. Train samples: {len(Xtr)}. Test samples: {len(Xte)}. Features: {len(Xtr.columns)}.")

    # 3. Train Models
    models: Dict[str, Any] = {}
    
    xgbm = train_xgb(Xtr, ytr, cfg)
    lgbm = train_lgb(Xtr, ytr, cfg)
    
    if lgbm: models['lgb'] = lgbm
    if xgbm: models['xgb'] = xgbm
    
    if not models: # Fallback to Ridge if no tree models were loaded
        ridgem = train_ridge_pipeline(Xtr, ytr, cfg)
        models['ridge'] = ridgem

    # 4. Test-time predictions & Blending
    pred_results: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        pred_results[name] = model.predict(Xte) 
    
    if 'xgb' in models and 'lgb' in models:
        log.info(f"Performing model blending/stacking with {cfg.stacker_model}...")
        
        model_preds = np.vstack([pred_results['xgb'], pred_results['lgb']])
        pred_results['median'] = np.median(model_preds, axis=0)
        
        # Split data for meta-model training (using internal validation set)
        X_blend_tr, X_blend_val, y_blend_tr, y_blend_val = train_test_split(
            Xtr, ytr, test_size=cfg.valid_size, shuffle=False, random_state=cfg.random_state
        )
        
        # Get base model predictions for the meta-model training (validation set) and test set
        X_meta_val = np.c_[models['xgb'].predict(X_blend_val), models['lgb'].predict(X_blend_val)]
        X_meta_te = np.c_[pred_results['xgb'], pred_results['lgb']]
        
        if cfg.stacker_model == 'Ridge':
            stacker = Ridge(alpha=cfg.stacker_alpha, positive=True, random_state=cfg.random_state)
        elif cfg.stacker_model == 'Lasso':
            stacker = Lasso(alpha=cfg.stacker_alpha, positive=True, random_state=cfg.random_state, max_iter=5000)
        elif cfg.stacker_model == 'RF':
            stacker = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=cfg.random_state, n_jobs=-1)
        else:
            stacker = Ridge(alpha=cfg.stacker_alpha, positive=True, random_state=cfg.random_state)

        stacker.fit(X_meta_val, y_blend_val)
        models['stacker'] = stacker 
        
        pred_results['stacked'] = stacker.predict(X_meta_te)
    
    # 5. Persistence baseline & Inverse Transform
    base_mets, y_true_level_untransformed, y_persist_untransformed = persistence_baseline(tr.loc[Xte.index] if len(tr.index.intersection(Xte.index)) > 0 else df_full.loc[Xte.index], yte, cfg.TARGET_COL, cfg)
    log.info(f"PERSISTENCE — MAE {base_mets['mae']:.3f} | RMSE {base_mets['rmse']:.3f} | R² {base_mets['r2']:.3f}")
    
    pred_results['persistence'] = y_persist_untransformed 

    pred_results = inverse_transform_predictions(pred_results, cfg)
    
    results = evaluate_heads(pred_results, y_true_level_untransformed, persist_rmse=base_mets["rmse"])
    
    # 6. Choose production
    candidates = [k for k in ['stacked', 'median', 'lgb', 'xgb', 'ridge'] if k in results]
    if not candidates:
        production_name = "persistence"
        log.warning("No model trained successfully. Shipping persistence.")
    else:
        best = max(candidates, key=lambda k: results[k].get("skill_vs_persist", -float('inf')))
        
        if results[best]["skill_vs_persist"] <= 0.0:
            production_name = "persistence"
            log.warning(f"No model beat persistence (best={best}, skill={results[best]['skill_vs_persist']:.3%}). Shipping persistence.")
        else:
            production_name = best
            log.info(f"Production head: {production_name} (skill={results[best]['skill_vs_persist']:.3%})")
            
    production_pred = pred_results[production_name]

    # 7. Save artifacts 
    log.info(f"Saving artifacts to {outdir.resolve()}")
    for name, model in models.items():
        joblib.dump(model, outdir / f"{name}_model_{stamp}.joblib")
    
    with open(outdir / f"config_{stamp}.json", 'w') as f:
        json.dump(asdict(cfg), f)
        
    with open(outdir / f"metrics_{stamp}.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    eval_index = Xte.index
    dfp = pd.DataFrame({
        "ts": eval_index, 
        "y_true": y_true_level_untransformed, 
        "y_persistence": pred_results["persistence"],
        "y_production": production_pred,
    }).set_index("ts")
    dfp.to_csv(outdir / f"predictions_{production_name}_{stamp}.csv")
    
    plot_predictions(eval_index, y_true_level_untransformed, production_pred, f"Production — {production_name}", outdir / f"plot_production_{stamp}.png")
    log.info("Done.")


# Run the training process
if __name__ == '__main__':
    # Initial configuration (uses the stable settings from the fix)
    cfg = TrainConfig(
        FULL_DATA_PATH="smart_manufacturing_data.csv", 
        outdir="artifacts/models_v2", 
        horizon_min=5, 
        valid_size=0.1, 
        target_transformation=False 
    )
    log.info(f"Config: {asdict(cfg)}")
    run(cfg)