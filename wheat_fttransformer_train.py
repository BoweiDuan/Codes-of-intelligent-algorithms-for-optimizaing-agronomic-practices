# -*- coding: utf-8 -*-
"""
Train and evaluate an FTTransformer surrogate model for wheat yield.

What this script does
- Loads the wheat dataset from an Excel file.
- Splits data into train/validation/test = 70%/10%/20%.
- Runs 10-fold cross-validation on the training split to estimate stability across folds.
- Trains a final model on the full training split with early stopping on the validation split.
- Evaluates the final model on the held-out test split.
- Repeats the entire pipeline for 10 random seeds and reports aggregated metrics.

Data note
- The original survey dataset is not included in this repository. Provide your own file with the same schema.
- Default path: data/wheat_filtered.xlsx (override with DATA_DIR and WHEAT_FILE env vars).

Outputs
- saved_wheat_yield_models_tmppre_ftt/seed_<seed>/best_model.pt
- saved_wheat_yield_models_tmppre_ftt/seed_<seed>/scaler_X.joblib
- saved_wheat_yield_models_tmppre_ftt/seed_<seed>/scaler_y.joblib
- saved_wheat_yield_models_tmppre_ftt/summary_metrics.csv
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import rtdl  # FTTransformer implementation


# -------------------------------
# Configuration
# -------------------------------

BASE_SEED = 123
N_SEEDS = 10
N_FOLDS = 10

SAVE_DIR = Path("saved_wheat_models_ftt")

DATA_DIR = Path(os.environ.get("DATA_DIR", "data_demo"))
WHEAT_FILE = os.environ.get("WHEAT_FILE", "wheat.xlsx")
FILE_PATH = DATA_DIR / WHEAT_FILE

# If your Excel uses a different yield column name, change it here.
TARGET_COL = "Yield"

FEATURE_COLS: List[str] = [
    "Seed",
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

# Filter used in your original script
MIN_YIELD_FILTER = 6000

# Training hyperparameters (keep conservative defaults; adjust if needed)
BATCH_SIZE = 128
MAX_EPOCHS = 1000
PATIENCE = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5


def get_device() -> torch.device:
    """Prefer XPU (if available), then CUDA, else CPU."""
    if hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None)) and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_seeds(base_seed: int = BASE_SEED, n: int = N_SEEDS) -> List[int]:
    """Generate a deterministic list of seeds."""
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

    # val is 10% of full; within dev (80%), val fraction is 0.1/0.8 = 0.125
    val_frac_of_dev = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_frac_of_dev, random_state=seed, shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Fit scalers on training split only; transform all splits."""
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).astype(np.float32)
    y_val_s = scaler_y.transform(y_val.to_numpy().reshape(-1, 1)).astype(np.float32)
    y_test_s = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).astype(np.float32)

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler_X, scaler_y


def build_model(n_features: int, seed: int) -> torch.nn.Module:
    """Create an FTTransformer model with fixed hyperparameters."""
    torch.manual_seed(seed)
    model = rtdl.FTTransformer.make_default(
        n_num_features=n_features,
        cat_cardinalities=None,
        d_out=1,
    )
    return model


@torch.no_grad()
def predict_scaled(model: torch.nn.Module, X_np: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X_np.astype(np.float32)).to(device)
    y_hat = model(X_t).detach().cpu().numpy().reshape(-1, 1)
    return y_hat


def train_one_model(
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_val_s: np.ndarray,
    y_val_s: np.ndarray,
    seed: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, float]:
    """Train with early stopping on the provided validation set. Returns (best_model, best_val_loss)."""
    set_seeds(seed)

    model = build_model(n_features=X_train_s.shape[1], seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
    y_train_t = torch.from_numpy(y_train_s.astype(np.float32))
    X_val_t = torch.from_numpy(X_val_s.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val_s.astype(np.float32)).to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        val_pred = model(X_val_t)
        val_loss = float(loss_fn(val_pred, y_val_t).item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def cv_on_training_split(
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    y_train_original: np.ndarray,
    scaler_y: StandardScaler,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    10-fold CV on the training split (70%).
    We standardize y and report metrics in original units.
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rmses, r2s = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_s), start=1):
        X_tr, X_va = X_train_s[tr_idx], X_train_s[va_idx]
        y_tr, y_va = y_train_s[tr_idx], y_train_s[va_idx]

        # Need original-unit y for the fold val set:
        y_va_original = y_train_original[va_idx]

        model, _ = train_one_model(X_tr, y_tr, X_va, y_va, seed=seed + fold, device=device)
        y_hat_va_scaled = predict_scaled(model, X_va, device=device)
        y_hat_va_original = scaler_y.inverse_transform(y_hat_va_scaled).reshape(-1)

        rmse = float(np.sqrt(mean_squared_error(y_va_original, y_hat_va_original)))
        r2 = float(r2_score(y_va_original, y_hat_va_original))

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
            f"Dataset not found: {path}. Set DATA_DIR and WHEAT_FILE to point to your data file."
        )
    df = pd.read_excel(path)
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)[:30]} ...")
    df = df.reset_index(drop=True)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Training device: {device}")
    print(f"Dataset: {FILE_PATH}")

    seeds = make_seeds()
    rows = []

    for i, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 80)
        print(f"[Wheat FTT] Run {i}/{len(seeds)} | seed={seed}")

        set_seeds(seed)

        X, y = load_dataset(FILE_PATH)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, seed=seed)

        (
            X_train_s,
            X_val_s,
            X_test_s,
            y_train_s,
            y_val_s,
            y_test_s,
            scaler_X,
            scaler_y,
        ) = standardize_splits(X_train, X_val, X_test, y_train, y_val, y_test)

        y_train_original = y_train.to_numpy().reshape(-1)
        y_val_original = y_val.to_numpy().reshape(-1)
        y_test_original = y_test.to_numpy().reshape(-1)

        cv_metrics = cv_on_training_split(
            X_train_s=X_train_s,
            y_train_s=y_train_s,
            y_train_original=y_train_original,
            scaler_y=scaler_y,
            seed=seed,
            device=device,
        )
        print(
            f"10-fold CV (train split): RMSE={cv_metrics['cv_rmse_mean']:.3f}±{cv_metrics['cv_rmse_std']:.3f}, "
            f"R2={cv_metrics['cv_r2_mean']:.3f}±{cv_metrics['cv_r2_std']:.3f}"
        )

        start = time.time()
        final_model, best_val_loss = train_one_model(
            X_train_s=X_train_s,
            y_train_s=y_train_s,
            X_val_s=X_val_s,
            y_val_s=y_val_s,
            seed=seed,
            device=device,
        )
        elapsed = time.time() - start

        # Final evaluation (original units)
        y_hat_train = scaler_y.inverse_transform(predict_scaled(final_model, X_train_s, device=device)).reshape(-1)
        y_hat_val = scaler_y.inverse_transform(predict_scaled(final_model, X_val_s, device=device)).reshape(-1)
        y_hat_test = scaler_y.inverse_transform(predict_scaled(final_model, X_test_s, device=device)).reshape(-1)

        train_rmse = float(np.sqrt(mean_squared_error(y_train_original, y_hat_train)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_original, y_hat_val)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test_original, y_hat_test)))

        train_r2 = float(r2_score(y_train_original, y_hat_train))
        val_r2 = float(r2_score(y_val_original, y_hat_val))
        test_r2 = float(r2_score(y_test_original, y_hat_test))

        print(f"Final training done in {elapsed:.1f}s | best_val_loss(scaled)={best_val_loss:.6f}")
        print(f"Train RMSE={train_rmse:.3f}, R2={train_r2:.3f}")
        print(f"Val   RMSE={val_rmse:.3f}, R2={val_r2:.3f}")
        print(f"Test  RMSE={test_rmse:.3f}, R2={test_r2:.3f}")

        run_dir = SAVE_DIR / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(final_model.state_dict(), run_dir / "best_model.pt")
        joblib.dump(scaler_X, run_dir / "scaler_X.joblib")
        joblib.dump(scaler_y, run_dir / "scaler_y.joblib")
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
                "best_val_loss_scaled": best_val_loss,
                "train_time_sec": elapsed,
            }
        )

    summary_path = SAVE_DIR / "summary_metrics.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print("\nSaved metrics summary:", summary_path)


if __name__ == "__main__":
    main()
