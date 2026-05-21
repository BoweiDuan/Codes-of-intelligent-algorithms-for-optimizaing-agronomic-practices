# -*- coding: utf-8 -*-
"""
Farmer typology identification via Ward.D2 clustering (three indicators).

Pipeline
1) Read the current-practice dataset for each crop (wheat, maize).
2) Load the trained surrogate model and predict baseline yield for each record.
3) Compute EI and EcI using the predicted baseline yield (not the observed yield).
4) Standardize (Yield_pred, EI, EcI) and run Ward clustering (3 clusters).
5) Export cluster assignments and optional diagnostic plots (elbow + dendrogram + scatter).

Data note
- The original survey dataset is not included in this repository.
- Provide your own Excel files with the required schema.

Default inputs (override via environment variables)
- Wheat: data_demo/wheat.xlsx  (WHEAT_CLUSTER_INPUT)
- Maize: data_demo/maize.xlsx  (MAIZE_CLUSTER_INPUT)

Default model directories (override via environment variables)
- Wheat FTT: models/saved_wheat_models_ftt (WHEAT_MODEL_DIR)
- Maize XGB: models/saved_maize_models_xgb (MAIZE_MODEL_DIR)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import rtdl

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from sklearn.preprocessing import StandardScaler


# -------------------------------
# Configuration (style aligned with your optimization scripts)
# -------------------------------

class CONFIG:
    SEED = 123

    # Inputs
    WHEAT_CLUSTER_INPUT = os.environ.get("WHEAT_CLUSTER_INPUT", "data_demo/wheat.xlsx")
    MAIZE_CLUSTER_INPUT = os.environ.get("MAIZE_CLUSTER_INPUT", "data_demo/maize.xlsx")

    # Model directories
    WHEAT_MODEL_DIR = os.environ.get("WHEAT_MODEL_DIR", "models/saved_wheat_models_ftt")
    MAIZE_MODEL_DIR = os.environ.get("MAIZE_MODEL_DIR", "models/saved_maize_models_xgb")

    # Outputs
    OUT_DIR = os.environ.get("CLUSTER_OUT_DIR", "results/typology_clustering")
    N_CLUSTERS = 3

    # Plotting
    CLUSTER_PALETTE_HEX = ["#E64B35", "#4DBBD5", "#00A087"]  # C1, C2, C3
    SAVE_PLOTS = True

    # Wheat features used by the surrogate model (must match training schema)
    WHEAT_MODEL_FEATURES: List[str] = [
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

    # Maize features used by the surrogate model (must match training schema)
    MAIZE_MODEL_FEATURES: List[str] = [
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


# -------------------------------
# Model loaders / predictors
# -------------------------------

class WheatYieldPredictor:
    def __init__(self, model_dir: str, device_str: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device_str) if device_str else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_order: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        scalerX_candidates = [
            self.model_dir / "scaler_X.joblib",
            self.model_dir / "scaler_X_for_ftt.joblib",
        ]
        scalery_candidates = [
            self.model_dir / "scaler_y.joblib",
            self.model_dir / "scaler_y_for_ftt.joblib",
        ]
        model_candidates = [
            self.model_dir / "best_model.pt",
            self.model_dir / "final_best_ftt_model.pt",
        ]

        scaler_X_path = next((p for p in scalerX_candidates if p.exists()), None)
        scaler_y_path = next((p for p in scalery_candidates if p.exists()), None)
        model_path = next((p for p in model_candidates if p.exists()), None)

        if scaler_X_path is None or scaler_y_path is None:
            raise FileNotFoundError(f"Missing FTT scalers in: {self.model_dir}")
        if model_path is None:
            raise FileNotFoundError(f"Missing FTT weights in: {self.model_dir}")

        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.feature_order = self.scaler_X.feature_names_in_

        n_features = int(len(self.feature_order))
        self.model = rtdl.FTTransformer.make_default(
            n_num_features=n_features,
            cat_cardinalities=None,
            d_out=1,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        df_ordered = df_features[self.feature_order].astype(float)
        X_scaled = self.scaler_X.transform(df_ordered)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_scaled = self.model(X_t, x_cat=None).detach().cpu().numpy().reshape(-1, 1)
        y = self.scaler_y.inverse_transform(y_scaled).reshape(-1)
        return y


class MaizeYieldPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model: xgb.XGBRegressor | None = None
        self.scaler_X = None
        self.feature_order: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        scaler_candidates = [
            self.model_dir / "scaler_X.joblib",
            self.model_dir / "scaler_for_xgb.joblib",
        ]
        model_candidates = [
            self.model_dir / "xgb_model.json",
            self.model_dir / "manual_best_xgb_model.json",
        ]
        scaler_path = next((p for p in scaler_candidates if p.exists()), None)
        model_path = next((p for p in model_candidates if p.exists()), None)

        if scaler_path is None:
            raise FileNotFoundError(f"Missing XGB scaler_X in: {self.model_dir}")
        if model_path is None:
            raise FileNotFoundError(f"Missing XGB model json in: {self.model_dir}")

        self.scaler_X = joblib.load(scaler_path)
        self.feature_order = self.scaler_X.feature_names_in_

        self.model = xgb.XGBRegressor(tree_method="hist", n_jobs=1)
        self.model.load_model(model_path)

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        df_ordered = df_features[self.feature_order].astype(float)
        X_scaled = self.scaler_X.transform(df_ordered)
        return self.model.predict(X_scaled).reshape(-1)


# -------------------------------
# EI / EcI calculators (baseline: predicted yield)
# -------------------------------

@dataclass(frozen=True)
class EI_EcI_Params:
    # Energy/emission coefficients
    CEC: Dict[str, float]
    NEC: Dict[str, float]
    KWH_PER_M3_WATER: float

    # Prices/costs
    PRICES: Dict[str, float]
    PRICE_WATER: float
    COST_MACHINERY: float
    COST_LABOR: float

    # Crop-specific extras
    AVG_SEED_WEIGHT_GRAM: float | None = None  # maize only


def _ensure_required_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in required:
        if c not in df2.columns:
            df2[c] = 0.0
    return df2


def _compute_growing_period(df: pd.DataFrame) -> pd.Series:
    if "HarvestDate" in df.columns and "SowDate" in df.columns:
        return pd.to_numeric(df["HarvestDate"], errors="coerce").fillna(0.0) - pd.to_numeric(df["SowDate"], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def compute_indices_wheat(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    p = EI_EcI_Params(
        CEC={"irrigation_power": 0.92, "N_fertilizer": 1.53, "P_fertilizer": 1.14, "K_fertilizer": 0.66, "pesticide": 6.58, "seed": 1.16},
        NEC={"irrigation_power": 0.12e-3, "N_fertilizer": 0.89e-3, "P_fertilizer": 0.54e-3, "K_fertilizer": 0.03e-3, "pesticide": 5.02e-3, "seed": 0.24e-3},
        KWH_PER_M3_WATER=1.0,
        PRICES={"grain": 0.35, "N_fertilizer": 0.7, "P_fertilizer": 0.86, "K_fertilizer": 0.86, "pesticide": 24.74, "seed": 0.6},
        PRICE_WATER=0.072,
        COST_MACHINERY=432.14,
        COST_LABOR=39.80,
    )

    req = ["Seed", "BasalN", "TopdressingN", "BasalP2O5", "TopdressingP2O5", "BasalK2O", "TopdressingK2O",
           "Irrigation1Amount", "Irrigation2Amount", "Irrigation3Amount", "Pesticide"]
    d = _ensure_required_columns(df, req)

    seed = pd.to_numeric(d["Seed"], errors="coerce").fillna(0.0).to_numpy()
    N = (pd.to_numeric(d["BasalN"], errors="coerce").fillna(0.0) + pd.to_numeric(d["TopdressingN"], errors="coerce").fillna(0.0)).to_numpy()
    P = (pd.to_numeric(d["BasalP2O5"], errors="coerce").fillna(0.0) + pd.to_numeric(d["TopdressingP2O5"], errors="coerce").fillna(0.0)).to_numpy()
    K = (pd.to_numeric(d["BasalK2O"], errors="coerce").fillna(0.0) + pd.to_numeric(d["TopdressingK2O"], errors="coerce").fillna(0.0)).to_numpy()
    TF = N + P + K
    IWR = (pd.to_numeric(d["Irrigation1Amount"], errors="coerce").fillna(0.0) +
           pd.to_numeric(d["Irrigation2Amount"], errors="coerce").fillna(0.0) +
           pd.to_numeric(d["Irrigation3Amount"], errors="coerce").fillna(0.0)).to_numpy()
    TP = pd.to_numeric(d["Pesticide"], errors="coerce").fillna(0.0).to_numpy()

    eps = 1e-6
    y = np.asarray(y_pred, dtype=float)

    ip = IWR * p.KWH_PER_M3_WATER
    ce = ip * p.CEC["irrigation_power"] + N * p.CEC["N_fertilizer"] + P * p.CEC["P_fertilizer"] + K * p.CEC["K_fertilizer"] + TP * p.CEC["pesticide"] + seed * p.CEC["seed"]
    ne_up = ip * p.NEC["irrigation_power"] + N * p.NEC["N_fertilizer"] + P * p.NEC["P_fertilizer"] + K * p.NEC["K_fertilizer"] + TP * p.NEC["pesticide"] + seed * p.NEC["seed"]

    n2o = N * 0.0105 * (44 / 28) * 0.476
    nh3 = N * 0.16 * (17 / 14) * 0.833
    no3 = N * 0.14 * (62 / 14) * 0.238
    ne = ne_up + n2o + nh3 + no3

    CI = ce / (y + eps)
    NI = ne / (y + eps)
    WF = IWR / (y + eps)

    z_ci = (CI.mean() - CI) / (CI.std() + eps)
    z_ni = (NI.mean() - NI) / (NI.std() + eps)
    z_wf = (WF.mean() - WF) / (WF.std() + eps)
    z_tf = (TF.mean() - TF) / (TF.std() + eps)
    z_iwr = (IWR.mean() - IWR) / (IWR.std() + eps)
    z_tp = (TP.mean() - TP) / (TP.std() + eps)
    EI = np.mean(np.vstack([z_ci, z_ni, z_wf, z_tf, z_iwr, z_tp]), axis=0)

    rev = y * p.PRICES["grain"]
    cost = (N * p.PRICES["N_fertilizer"] + P * p.PRICES["P_fertilizer"] + K * p.PRICES["K_fertilizer"] + TP * p.PRICES["pesticide"] + seed * p.PRICES["seed"]) + IWR * p.PRICE_WATER + p.COST_MACHINERY + p.COST_LABOR
    prof = rev - cost
    roi = prof / (cost + eps)

    z_prof = (prof - prof.mean()) / (prof.std() + eps)
    z_roi = (roi - roi.mean()) / (roi.std() + eps)
    EcI = np.mean(np.vstack([z_prof, z_roi]), axis=0)

    out = df.copy()
    out["Yield_Baseline_Pred"] = y
    out["EI_Baseline"] = EI
    out["EcI_Baseline"] = EcI
    return out


def compute_indices_maize(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    p = EI_EcI_Params(
        CEC={"irrigation_power": 0.92, "N_fertilizer": 1.53, "P_fertilizer": 1.14, "K_fertilizer": 0.66, "pesticide": 6.58, "seed": 1.22},
        NEC={"irrigation_power": 0.12e-3, "N_fertilizer": 0.89e-3, "P_fertilizer": 0.54e-3, "K_fertilizer": 0.03e-3, "pesticide": 5.02e-3, "seed": 0.88e-3},
        KWH_PER_M3_WATER=1.0,
        PRICES={"grain": 0.35, "N_fertilizer": 0.7, "P_fertilizer": 0.86, "K_fertilizer": 0.86, "pesticide": 24.74, "seed": 5.57},
        PRICE_WATER=0.072,
        COST_MACHINERY=151.06,
        COST_LABOR=73.09,
        AVG_SEED_WEIGHT_GRAM=344.0,
    )

    req = ["Density", "BasalN", "TopdressingN", "BasalP2O5", "TopdressingP2O5", "BasalK2O", "TopdressingK2O",
           "Irrigation1Amount", "Irrigation2Amount", "Irrigation3Amount", "Pesticide"]
    d = _ensure_required_columns(df, req)

    den = pd.to_numeric(d["Density"], errors="coerce").fillna(0.0).to_numpy()
    sw = den * float(p.AVG_SEED_WEIGHT_GRAM) / 1000000.0
    N = (pd.to_numeric(d["BasalN"], errors="coerce").fillna(0.0) + pd.to_numeric(d["TopdressingN"], errors="coerce").fillna(0.0)).to_numpy()
    P = (pd.to_numeric(d["BasalP2O5"], errors="coerce").fillna(0.0) + pd.to_numeric(d["TopdressingP2O5"], errors="coerce").fillna(0.0)).to_numpy()
    K = (pd.to_numeric(d["BasalK2O"], errors="coerce").fillna(0.0) + pd.to_numeric(d["TopdressingK2O"], errors="coerce").fillna(0.0)).to_numpy()
    TF = N + P + K
    IWR = (pd.to_numeric(d["Irrigation1Amount"], errors="coerce").fillna(0.0) +
           pd.to_numeric(d["Irrigation2Amount"], errors="coerce").fillna(0.0) +
           pd.to_numeric(d["Irrigation3Amount"], errors="coerce").fillna(0.0)).to_numpy()
    TP = pd.to_numeric(d["Pesticide"], errors="coerce").fillna(0.0).to_numpy()

    eps = 1e-6
    y = np.asarray(y_pred, dtype=float)

    ip = IWR * p.KWH_PER_M3_WATER
    ce = ip * p.CEC["irrigation_power"] + N * p.CEC["N_fertilizer"] + P * p.CEC["P_fertilizer"] + K * p.CEC["K_fertilizer"] + TP * p.CEC["pesticide"] + sw * p.CEC["seed"]
    ne_up = ip * p.NEC["irrigation_power"] + N * p.NEC["N_fertilizer"] + P * p.NEC["P_fertilizer"] + K * p.NEC["K_fertilizer"] + TP * p.NEC["pesticide"] + sw * p.NEC["seed"]

    n2o = N * 0.0250 * (44 / 28) * 0.476
    nh3 = N * 0.21 * (17 / 14) * 0.833
    no3 = N * 0.19 * (62 / 14) * 0.238
    ne = ne_up + n2o + nh3 + no3

    CI = ce / (y + eps)
    NI = ne / (y + eps)
    WF = IWR / (y + eps)

    z_ci = (CI.mean() - CI) / (CI.std() + eps)
    z_ni = (NI.mean() - NI) / (NI.std() + eps)
    z_wf = (WF.mean() - WF) / (WF.std() + eps)
    z_tf = (TF.mean() - TF) / (TF.std() + eps)
    z_iwr = (IWR.mean() - IWR) / (IWR.std() + eps)
    z_tp = (TP.mean() - TP) / (TP.std() + eps)
    EI = np.mean(np.vstack([z_ci, z_ni, z_wf, z_tf, z_iwr, z_tp]), axis=0)

    rev = y * p.PRICES["grain"]
    cost = (N * p.PRICES["N_fertilizer"] + P * p.PRICES["P_fertilizer"] + K * p.PRICES["K_fertilizer"] + TP * p.PRICES["pesticide"] + sw * p.PRICES["seed"]) + IWR * p.PRICE_WATER + p.COST_MACHINERY + p.COST_LABOR
    prof = rev - cost
    roi = prof / (cost + eps)

    z_prof = (prof - prof.mean()) / (prof.std() + eps)
    z_roi = (roi - roi.mean()) / (roi.std() + eps)
    EcI = np.mean(np.vstack([z_prof, z_roi]), axis=0)

    out = df.copy()
    out["Yield_Baseline_Pred"] = y
    out["EI_Baseline"] = EI
    out["EcI_Baseline"] = EcI
    return out


# -------------------------------
# Clustering + plotting
# -------------------------------

def plot_diagnostics(Z: np.ndarray, title: str, out_dir: Path) -> None:
    if not CONFIG.SAVE_PLOTS:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Elbow (Ward distances)
    plt.figure(figsize=(6, 4))
    last = Z[-15:, 2][::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last, "o-", linewidth=2, markersize=6)
    plt.title(f"{title} clustering - elbow (Ward)", fontweight="bold", fontsize=12)
    plt.xlabel("Number of clusters (proxy)")
    plt.ylabel("Merge distance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}_Elbow.svg", format="svg", bbox_inches="tight")
    plt.close()

    # Dendrogram (truncated)
    plt.figure(figsize=(8, 5))
    set_link_color_palette(CONFIG.CLUSTER_PALETTE_HEX)
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90.0,
        leaf_font_size=10.0,
        show_contracted=True,
        color_threshold=(Z[-CONFIG.N_CLUSTERS, 2] + Z[-(CONFIG.N_CLUSTERS - 1), 2]) / 2,
        above_threshold_color="#AAAAAA",
    )
    plt.title(f"{title} clustering - dendrogram", fontweight="bold", fontsize=12)
    plt.xlabel("Cluster size / sample index")
    plt.ylabel("Distance")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}_Dendrogram.svg", format="svg", bbox_inches="tight")
    plt.close()


def relabel_clusters_by_positions(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Map raw cluster ids (1..K) into stable labels based on relative positions:
    - C1: high yield and high EI
    - C2: low EI
    - C3: high EI but low yield
    """
    df2 = df.copy()
    raw_ids = sorted(df2["Raw_Cluster"].unique().tolist())

    centers = []
    for cid in raw_ids:
        sub = df2[df2["Raw_Cluster"] == cid]
        centers.append((cid, float(sub[x_col].mean()), float(sub[y_col].mean())))

    x_global = float(df2[x_col].mean())
    y_global = float(df2[y_col].mean())

    # heuristic assignment
    c1 = max(centers, key=lambda t: (t[2] - y_global) + (t[1] - x_global))[0]
    remaining = [t for t in centers if t[0] != c1]
    c2 = min(remaining, key=lambda t: (t[1] - x_global))[0]  # lowest EI
    c3 = [t[0] for t in remaining if t[0] != c2][0]

    label_map = {
        c1: "C1 (High Synergy)",
        c2: "C2 (Low Efficiency)",
        c3: "C3 (Env-oriented)",
    }
    df2["Cluster_Label"] = df2["Raw_Cluster"].map(label_map)
    return df2, label_map


def plot_scatter(df: pd.DataFrame, title: str, out_dir: Path, x_col: str, y_col: str) -> None:
    if not CONFIG.SAVE_PLOTS:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    for i, lab in enumerate(["C1 (High Synergy)", "C2 (Low Efficiency)", "C3 (Env-oriented)"]):
        sub = df[df["Cluster_Label"] == lab]
        if len(sub) == 0:
            continue
        ax.scatter(sub[x_col], sub[y_col], s=50, alpha=0.85, label=lab, color=CONFIG.CLUSTER_PALETTE_HEX[i])

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_')}.svg", format="svg", bbox_inches="tight")
    plt.close()


def cluster_one_crop(df: pd.DataFrame, crop_name: str, out_dir: Path) -> pd.DataFrame:
    features = ["Yield_Baseline_Pred", "EI_Baseline", "EcI_Baseline"]
    df_clean = df.dropna(subset=features).copy()

    X = df_clean[features].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    Z = linkage(X_scaled, method="ward", metric="euclidean")
    plot_diagnostics(Z, crop_name, out_dir)

    df_clean["Raw_Cluster"] = fcluster(Z, t=CONFIG.N_CLUSTERS, criterion="maxclust")
    df_labeled, _ = relabel_clusters_by_positions(df_clean, x_col="EI_Baseline", y_col="Yield_Baseline_Pred")
    plot_scatter(df_labeled, f"{crop_name} typology (baseline)", out_dir, x_col="EI_Baseline", y_col="Yield_Baseline_Pred")
    return df_labeled


# -------------------------------
# Main
# -------------------------------

def load_excel_numeric(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_excel(p)
    # Keep original columns, but coerce numerics where possible (do not drop rows here).
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    # GrowingPeriod is derived
    df["GrowingPeriod"] = _compute_growing_period(df)
    return df


def prepare_features_for_model(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    df2 = _ensure_required_columns(df, required_cols)
    # Ensure numeric for required columns
    for c in required_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)
    df2["GrowingPeriod"] = _compute_growing_period(df2)
    return df2


def main() -> None:
    np.random.seed(CONFIG.SEED)

    out_dir = Path(CONFIG.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wheat
    wheat_df_raw = load_excel_numeric(CONFIG.WHEAT_CLUSTER_INPUT)
    wheat_pred = WheatYieldPredictor(CONFIG.WHEAT_MODEL_DIR)
    wheat_feat = prepare_features_for_model(wheat_df_raw, CONFIG.WHEAT_MODEL_FEATURES)
    wheat_y = wheat_pred.predict(wheat_feat)
    wheat_df = compute_indices_wheat(wheat_df_raw, wheat_y)
    wheat_clustered = cluster_one_crop(wheat_df, "Wheat", out_dir)
    wheat_clustered.to_excel(out_dir / "wheat_typology_baseline.xlsx", index=False)

    # Maize
    maize_df_raw = load_excel_numeric(CONFIG.MAIZE_CLUSTER_INPUT)
    maize_pred = MaizeYieldPredictor(CONFIG.MAIZE_MODEL_DIR)
    maize_feat = prepare_features_for_model(maize_df_raw, CONFIG.MAIZE_MODEL_FEATURES)
    maize_y = maize_pred.predict(maize_feat)
    maize_df = compute_indices_maize(maize_df_raw, maize_y)
    maize_clustered = cluster_one_crop(maize_df, "Maize", out_dir)
    maize_clustered.to_excel(out_dir / "maize_typology_baseline.xlsx", index=False)

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
