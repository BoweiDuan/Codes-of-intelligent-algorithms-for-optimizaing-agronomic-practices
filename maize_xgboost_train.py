# -*- coding: utf-8 -*-
"""
Train and evaluate an XGBoost surrogate model for maize yield.

What this script does
- Loads the maize dataset from an Excel file.
- Splits data into train/validation/test = 70%/10%/20%.
- Runs 10-fold cross-validation on the training split to estimate stability across folds.
- Trains a final model on the full training split with early stopping on the validation split.
- Evaluates the final model on the held-out test split.
- Repeats the entire pipeline for 10 random seeds and reports aggregated metrics.

Data note
- The original survey dataset is not included in this repository. Provide your own file with the same schema.
- Default path: data_demo/maize.xlsx (override with DATA_DIR and MAIZE_FILE env vars).

Outputs
- models/saved_maize_models_xgb/seed_<seed>/xgb_model.json
- models/saved_maize_models_xgb/seed_<seed>/scaler_X.joblib
- models/saved_maize_models_xgb/summary_metrics.csv
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------
# Configuration
# -------------------------------

BASE_SEED = 123
N_SEEDS = 10
N_FOLDS = 10

SAVE_DIR = Path("models/saved_maize_models_xgb")

DATA_DIR = Path(os.environ.get("DATA_DIR", "data_demo"))
MAIZE_FILE = os.environ.get("MAIZE_FILE", "maize.xlsx")
FILE_PATH = DATA_DIR / MAIZE_FILE

TARGET_COL = "Yield"

FEATURE_COLS: List[str] = [
    "Density",
    "SowDate",
    "HarvestDate",
    "GrowingPeriod",
    "BasalN",
    "BasalP2O5",
    "BasalK2O",
    "Irrigation1Amount",
    "Irrigation2Amount",
    "Irrigation3Amount",
    "TopdressingN",
    "TopdressingP2O5",
    "TopdressingK2O",
    "Pesticide",
    "AP",
    "AK",
    "TN",
    "SOC",
    "pH",
    "Bulk",
    "SILT",
    "SAND",
    "CLAY",
    "atmp",
    "pre",
]


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def make_seeds(base_seed: int = BASE_SEED, n: int = N_SEEDS) -> List[int]:
    rng = np.random.RandomState(base_seed)
    return [int(s) for s in rng.randint(0, 10**9, size=n)]


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train/val/test (70/10/20) using simple random splits.

    Note: We intentionally avoid stratified sampling to prevent confusion about forcing
    distributional alignment across splits. Set the random seed for reproducibility.
    """
    assert abs(test_size - 0.2) < 1e-9
    assert abs(val_size - 0.1) < 1e-9

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True
    )

    val_frac_of_dev = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_frac_of_dev, random_state=seed, shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_X(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler_X


def build_model(seed: int) -> xgb.XGBRegressor:
    # Keep your original defaults; adjust if needed.
    params = {
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_estimators": 5000,
        "max_depth": 2,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "n_jobs": -1,
    }
    return xgb.XGBRegressor(**params)


def cv_on_training_split(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rmses, r2s = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_s), start=1):
        X_tr, X_va = X_train_s[tr_idx], X_train_s[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = build_model(seed + fold)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=200,
        )

        y_hat = model.predict(X_va)
        rmse = float(np.sqrt(mean_squared_error(y_va, y_hat)))
        r2 = float(r2_score(y_va, y_hat))
        rmses.append(rmse)
        r2s.append(r2)

    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "cv_r2_mean": float(np.mean(r2s)),
        "cv_r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
    }


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Set DATA_DIR and MAIZE_FILE to point to your data file."
        )
    df = pd.read_excel(path)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)[:30]} ...")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {FILE_PATH}")

    seeds = make_seeds()
    rows = []

    for i, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 80)
        print(f"[Maize XGB] Run {i}/{len(seeds)} | seed={seed}")

        set_seeds(seed)

        X, y = load_dataset(FILE_PATH)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, seed=seed)

        X_train_s, X_val_s, X_test_s, scaler_X = standardize_X(X_train, X_val, X_test)

        y_train_np = y_train.to_numpy()
        y_val_np = y_val.to_numpy()
        y_test_np = y_test.to_numpy()

        cv_metrics = cv_on_training_split(X_train_s, y_train_np, seed=seed)
        print(
            f"10-fold CV (train split): RMSE={cv_metrics['cv_rmse_mean']:.3f}±{cv_metrics['cv_rmse_std']:.3f}, "
            f"R2={cv_metrics['cv_r2_mean']:.3f}±{cv_metrics['cv_r2_std']:.3f}"
        )

        start = time.time()
        model = build_model(seed)
        model.fit(
            X_train_s,
            y_train_np,
            eval_set=[(X_val_s, y_val_np)],
            verbose=False,
            early_stopping_rounds=200,
        )
        elapsed = time.time() - start

        y_hat_train = model.predict(X_train_s)
        y_hat_val = model.predict(X_val_s)
        y_hat_test = model.predict(X_test_s)

        train_rmse = float(np.sqrt(mean_squared_error(y_train_np, y_hat_train)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_np, y_hat_val)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test_np, y_hat_test)))

        train_r2 = float(r2_score(y_train_np, y_hat_train))
        val_r2 = float(r2_score(y_val_np, y_hat_val))
        test_r2 = float(r2_score(y_test_np, y_hat_test))

        print(f"Final training done in {elapsed:.1f}s | best_iteration={model.best_iteration}")
        print(f"Train RMSE={train_rmse:.3f}, R2={train_r2:.3f}")
        print(f"Val   RMSE={val_rmse:.3f}, R2={val_r2:.3f}")
        print(f"Test  RMSE={test_rmse:.3f}, R2={test_r2:.3f}")

        run_dir = SAVE_DIR / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(run_dir / "xgb_model.json")
        joblib.dump(scaler_X, run_dir / "scaler_X.joblib")
        joblib.dump(FEATURE_COLS, run_dir / "feature_cols.joblib")

        rows.append(
            {
                "seed": seed,
                **cv_metrics,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
                "train_time_sec": elapsed,
            }
        )

    summary_path = SAVE_DIR / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print("\nSaved metrics summary:", summary_path)


if __name__ == "__main__":
    main()
