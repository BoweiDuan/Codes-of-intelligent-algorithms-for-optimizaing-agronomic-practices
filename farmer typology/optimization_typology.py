# -*- coding: utf-8 -*-
"""
Post-optimization clustering and novelty detection with Sankey visualization.

Typology definition (baseline):
- C1: Unsustainable
- C2: Efficiency
- C3: Environment-friendly

Novel cluster (post-optimization):
- C4: High-sustainability (flagged among solutions classified as C2 but sufficiently distant
      from the baseline C2 centroid and directionally better in standardized space).

Pipeline
1) Load optimization outputs (merged rows that include baseline inputs and optimized objectives).
2) Load trained surrogate models and recompute baseline yield for each record.
3) Recompute baseline EI and EcI using the same equations as the optimization code, using model-predicted baseline yield.
4) Cluster baseline points via Ward hierarchical clustering (Ward.D2) (K=3) in standardized space and relabel clusters to C1/C2/C3 via centroid rules.
5) Classify optimized solutions using KNN trained on baseline clusters (in the same standardized space).
6) Identify C4 among C2-classified optimized points using a distance threshold relative to baseline cluster separation.
7) Plot a two-panel Sankey figure (wheat and maize).

Inputs (defaults; override via environment variables)
- results/wheat_multiobjective/optimized_wheat_multiobjective.csv
- results/maize_multiobjective/optimized_maize_multiobjective.csv

Model artifacts (defaults; override via environment variables)
- Wheat (FTTransformer): models/saved_wheat_models_ftt
- Maize (XGBoost):      models/saved_maize_models_xgb
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean


# -------------------------------
# Configuration
# -------------------------------

class CONFIG:
    SEED = 123

    WHEAT_OPT_CSV = os.environ.get(
        "WHEAT_OPT_CSV",
        "results/wheat_multiobjective/optimized_wheat_multiobjective.csv",
    )
    MAIZE_OPT_CSV = os.environ.get(
        "MAIZE_OPT_CSV",
        "results/maize_multiobjective/optimized_maize_multiobjective.csv",
    )

    WHEAT_MODEL_DIR = os.environ.get("WHEAT_MODEL_DIR", "models/saved_wheat_models_ftt")
    MAIZE_MODEL_DIR = os.environ.get("MAIZE_MODEL_DIR", "models/saved_maize_models_xgb")

    OUT_DIR = os.environ.get("SANKEY_OUT_DIR", "results/figures")

    # Novelty detection parameters
    # KNN for post-optimization classification
    KNN_K = 5

    # Visualization
    FONT_FAMILY = os.environ.get("FIG_FONT_FAMILY", "Calibri")

    COLOR_MAP = {
        "C1: Unsustainable": "#A64036",
        "C2: Efficiency": "#3C6E9C",
        "C3: Env_friendly": "#387056",
        "C4: High-sustainability": "#E69F00",
        "Still C1": "#A64036",
        "Still C3": "#387056",
        "Standard C2": "#3C6E9C",
    }


# -------------------------------
# Model-based baseline yield
# -------------------------------

def _add_growing_period(df: pd.DataFrame) -> pd.DataFrame:
    if "GrowingPeriod" not in df.columns:
        if "HarvestDate" in df.columns and "SowDate" in df.columns:
            df["GrowingPeriod"] = df["HarvestDate"] - df["SowDate"]
        else:
            df["GrowingPeriod"] = 0.0
    return df


def _ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


class WheatYieldPredictor:
    def __init__(self, model_dir: str):
        import torch
        import rtdl

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_dir = Path(model_dir)
        scaler_x_path = next((p for p in [model_dir / "scaler_X.joblib", model_dir / "scaler_X_for_ftt.joblib"] if p.exists()), None)
        scaler_y_path = next((p for p in [model_dir / "scaler_y.joblib", model_dir / "scaler_y_for_ftt.joblib"] if p.exists()), None)
        model_path = next((p for p in [model_dir / "best_model.pt", model_dir / "final_best_ftt_model.pt"] if p.exists()), None)

        if scaler_x_path is None or scaler_y_path is None or model_path is None:
            raise FileNotFoundError(
                f"Missing wheat model artifacts under {model_dir} "
                f"(expected current or legacy scaler/model filenames)"
            )

        self.scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        self.feature_order = list(self.scaler_x.feature_names_in_)
        self.y_scale = torch.from_numpy(scaler_y.scale_).float().to(self.device)
        self.y_mean = torch.from_numpy(scaler_y.mean_).float().to(self.device)

        self.model = rtdl.FTTransformer.make_default(
            n_num_features=len(self.feature_order),
            cat_cardinalities=None,
            d_out=1,
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        torch = self.torch
        df = _add_growing_period(df_features.copy())
        df = _ensure_numeric(df, self.feature_order)
        X = self.scaler_x.transform(df[self.feature_order].astype(float))
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_scaled = self.model(X_t, x_cat=None)
            y = (y_scaled * self.y_scale + self.y_mean).detach().cpu().numpy().reshape(-1)
        return y


class MaizeYieldPredictor:
    def __init__(self, model_dir: str):
        import xgboost as xgb

        model_dir = Path(model_dir)
        scaler_x_path = next((p for p in [model_dir / "scaler_X.joblib", model_dir / "scaler_for_xgb.joblib"] if p.exists()), None)
        model_path = next((p for p in [model_dir / "xgb_model.json", model_dir / "manual_best_xgb_model.json"] if p.exists()), None)

        if scaler_x_path is None or model_path is None:
            raise FileNotFoundError(
                f"Missing maize model artifacts under {model_dir} "
                f"(expected current or legacy scaler/model filenames)"
            )

        self.scaler_x = joblib.load(scaler_x_path)
        self.feature_order = list(self.scaler_x.feature_names_in_)

        self.model = xgb.XGBRegressor(tree_method="hist", n_jobs=1)
        self.model.load_model(str(model_path))

    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        df = _add_growing_period(df_features.copy())
        df = _ensure_numeric(df, self.feature_order)
        X = self.scaler_x.transform(df[self.feature_order].astype(float))
        return self.model.predict(X).reshape(-1)


# -------------------------------
# Index calculator (EI, EcI)
# -------------------------------

@dataclass(frozen=True)
class CropParams:
    CEC: Dict[str, float]
    NEC: Dict[str, float]
    PRICES: Dict[str, float]
    COST_MACHINERY: float
    COST_LABOR: float
    PRICE_WATER: float = 0.072
    KWH_PER_M3_WATER: float = 1.0
    AVG_SEED_WEIGHT_GRAM: float | None = None  # Maize only


class MetricCalculator:
    @staticmethod
    def _get_params(crop_name: str) -> CropParams:
        if crop_name == "Maize":
            return CropParams(
                CEC={"irrigation_power": 0.92, "N_fertilizer": 1.53, "P_fertilizer": 1.14,
                     "K_fertilizer": 0.66, "pesticide": 6.58, "seed": 1.22},
                NEC={"irrigation_power": 0.12e-3, "N_fertilizer": 0.89e-3, "P_fertilizer": 0.54e-3,
                     "K_fertilizer": 0.03e-3, "pesticide": 5.02e-3, "seed": 0.88e-3},
                PRICES={"grain": 0.35, "N": 0.7, "P": 0.86, "K": 0.86, "pest": 24.74, "seed": 5.57, "water": 0.072},
                COST_MACHINERY=151.06,
                COST_LABOR=73.09,
                AVG_SEED_WEIGHT_GRAM=344.0,
            )
        if crop_name == "Wheat":
            return CropParams(
                CEC={"irrigation_power": 0.92, "N_fertilizer": 1.53, "P_fertilizer": 1.14,
                     "K_fertilizer": 0.66, "pesticide": 6.58, "seed": 1.16},
                NEC={"irrigation_power": 0.12e-3, "N_fertilizer": 0.89e-3, "P_fertilizer": 0.54e-3,
                     "K_fertilizer": 0.03e-3, "pesticide": 5.02e-3, "seed": 0.24e-3},
                PRICES={"grain": 0.35, "N": 0.7, "P": 0.86, "K": 0.86, "pest": 24.74, "seed": 0.6, "water": 0.072},
                COST_MACHINERY=432.14,
                COST_LABOR=39.80,
            )
        raise ValueError(f"Unknown crop: {crop_name}")

    @classmethod
    def compute_baseline_metrics(cls, df: pd.DataFrame, crop_name: str, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Compute baseline EI and EcI using model-predicted baseline yield (y_pred).
        """
        df = df.copy()
        p = cls._get_params(crop_name)

        eps = 1e-6
        yld = pd.Series(y_pred, index=df.index).astype(float)

        # Required inputs (missing values are set to 0.0)
        req = [
            "Density", "Seed",
            "BasalN", "TopdressingN",
            "BasalP2O5", "TopdressingP2O5",
            "BasalK2O", "TopdressingK2O",
            "Irrigation1Amount", "Irrigation2Amount", "Irrigation3Amount",
            "Pesticide",
        ]
        for c in req:
            if c not in df.columns:
                df[c] = 0.0
        df = _ensure_numeric(df, req)

        if crop_name == "Maize" and p.AVG_SEED_WEIGHT_GRAM is not None:
            sw = df["Density"] * p.AVG_SEED_WEIGHT_GRAM / 1000000.0
        else:
            sw = df["Seed"]

        N = df["BasalN"] + df["TopdressingN"]
        P = df["BasalP2O5"] + df["TopdressingP2O5"]
        K = df["BasalK2O"] + df["TopdressingK2O"]
        TF = N + P + K
        IWR = df["Irrigation1Amount"] + df["Irrigation2Amount"] + df["Irrigation3Amount"]
        TP = df["Pesticide"]

        ip = IWR * p.KWH_PER_M3_WATER

        ce = (
            ip * p.CEC["irrigation_power"]
            + N * p.CEC["N_fertilizer"]
            + P * p.CEC["P_fertilizer"]
            + K * p.CEC["K_fertilizer"]
            + TP * p.CEC["pesticide"]
            + sw * p.CEC["seed"]
        )

        ne_up = (
            ip * p.NEC["irrigation_power"]
            + N * p.NEC["N_fertilizer"]
            + P * p.NEC["P_fertilizer"]
            + K * p.NEC["K_fertilizer"]
            + TP * p.NEC["pesticide"]
            + sw * p.NEC["seed"]
        )

        if crop_name == "Maize":
            n2o = N * 0.0250 * (44 / 28) * 0.476
            nh3 = N * 0.21 * (17 / 14) * 0.833
            no3 = N * 0.19 * (62 / 14) * 0.238
        else:
            n2o = N * 0.0105 * (44 / 28) * 0.476
            nh3 = N * 0.16 * (17 / 14) * 0.833
            no3 = N * 0.14 * (62 / 14) * 0.238

        ne = ne_up + n2o + nh3 + no3

        CI = ce / (yld + eps)
        NI = ne / (yld + eps)
        WF = IWR / (yld + eps)

        stats = [CI, NI, WF, TF, IWR, TP]
        z_scores = [(c.mean() - c) / (c.std() + eps) for c in stats]
        ei = np.mean(np.vstack(z_scores), axis=0)

        rev = yld * p.PRICES["grain"]
        cost = (
            N * p.PRICES["N"]
            + P * p.PRICES["P"]
            + K * p.PRICES["K"]
            + TP * p.PRICES["pest"]
            + sw * p.PRICES["seed"]
            + IWR * p.PRICES["water"]
            + p.COST_MACHINERY
            + p.COST_LABOR
        )
        prof = rev - cost
        roi = prof / (cost + eps)

        z_prof = (prof - prof.mean()) / (prof.std() + eps)
        z_roi = (roi - roi.mean()) / (roi.std() + eps)
        eci = np.mean(np.vstack([z_prof, z_roi]), axis=0)

        return pd.DataFrame({"Yield": yld.to_numpy(), "EI": ei, "EcI": eci}, index=df.index)


# -------------------------------
# Clustering and novelty detection
# -------------------------------

def assign_baseline_clusters(df_metrics: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, Dict[int, str], np.ndarray, float]:
    """
    Ward hierarchical clustering (Ward.D2, k=3) in standardized space; relabel clusters to typologies using centroid rules.

    - C2 (Efficiency): highest centroid sum in standardized space.
    - C1 (Unsustainable): lowest centroid sum among remaining clusters.
    - C3 (Env_friendly): the remaining cluster.

    Note: Ward.D2 corresponds to Ward linkage with squared Euclidean distance (variance-minimizing).
    """
    X = df_metrics[["Yield", "EI", "EcI"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
    raw = ward.fit_predict(X_std)

    centroids_std = np.vstack([X_std[raw == k].mean(axis=0) for k in range(3)])
    sums = centroids_std.sum(axis=1)

    c2_idx = int(np.argmax(sums))
    remaining = [i for i in range(3) if i != c2_idx]
    c1_idx = int(remaining[int(np.argmin(sums[remaining]))])
    c3_idx = int([i for i in remaining if i != c1_idx][0])

    label_map = {
        c1_idx: "C1: Unsustainable",
        c2_idx: "C2: Efficiency",
        c3_idx: "C3: Env_friendly",
    }
    labels = np.array([label_map[int(k)] for k in raw], dtype=object)

    c_std = {name: X_std[labels == name].mean(axis=0) for name in label_map.values()}
    dists = [
        euclidean(c_std["C1: Unsustainable"], c_std["C2: Efficiency"]),
        euclidean(c_std["C1: Unsustainable"], c_std["C3: Env_friendly"]),
        euclidean(c_std["C2: Efficiency"], c_std["C3: Env_friendly"]),
    ]
    thr = float(min(dists))
    return labels, scaler, label_map, c_std["C2: Efficiency"], thr
def identify_structural_evolution(
    df_full: pd.DataFrame,
    crop_name: str,
    baseline_yield_pred: np.ndarray,
) -> pd.DataFrame:
    calc = MetricCalculator()

    df_base = calc.compute_baseline_metrics(df_full, crop_name, baseline_yield_pred)
    y_base_labels, scaler, _, c2_centroid, threshold_dist = assign_baseline_clusters(df_base)

    X_base_std = scaler.transform(df_base[["Yield", "EI", "EcI"]].to_numpy(dtype=float))

    opt_cols = ["Objective_Yield", "Objective_Environmental_Index", "Objective_Economic_Index"]
    for c in opt_cols:
        if c not in df_full.columns:
            raise KeyError(f"Missing required optimized column: {c}")

    X_opt = df_full[opt_cols].to_numpy(dtype=float)
    X_opt_std = scaler.transform(X_opt)

    knn = KNeighborsClassifier(n_neighbors=int(CONFIG.KNN_K))
    knn.fit(X_base_std, y_base_labels)
    y_pred_labels = knn.predict(X_opt_std)

    final_targets = []
    for i, lab in enumerate(y_pred_labels):
        if lab == "C2: Efficiency":
            d = euclidean(X_opt_std[i], c2_centroid)
            is_better = float(np.sum(X_opt_std[i])) > float(np.sum(c2_centroid))
            if d > threshold_dist and is_better:
                final_targets.append("C4: High-sustainability")
            else:
                final_targets.append("Standard C2")
        elif lab == "C1: Unsustainable":
            final_targets.append("Still C1")
        else:
            final_targets.append("Still C3")

    return pd.DataFrame({"Source": y_base_labels, "Target": np.array(final_targets, dtype=object)})


# -------------------------------
# Sankey plotting (matplotlib)
# -------------------------------

def plot_sankey(ax, df_flow: pd.DataFrame, title: str):
    counts = df_flow.groupby(["Source", "Target"]).size().reset_index(name="Count")
    total = int(counts["Count"].sum())
    if total == 0:
        ax.axis("off")
        ax.set_title(title)
        return

    LEFT, RIGHT = 0.1, 0.9
    WIDTH, GAP = 0.05, 0.08

    src_order = ["C1: Unsustainable", "C2: Efficiency", "C3: Env_friendly"]
    tgt_order = ["C4: High-sustainability", "Standard C2", "Still C3", "Still C1"]

    src_order = [s for s in src_order if s in df_flow["Source"].unique()]
    tgt_order = [t for t in tgt_order if t in df_flow["Target"].unique()]

    src_stats = df_flow["Source"].value_counts(normalize=True).reindex(src_order, fill_value=0.0)
    tgt_stats = df_flow["Target"].value_counts(normalize=True).reindex(tgt_order, fill_value=0.0)

    y_cursor = 1.0
    src_coords = {}
    valid_h = 1.0 - (len(src_order) - 1) * GAP
    for src in src_order:
        h = float(src_stats[src] * valid_h)
        h = max(h, 0.002)
        color = CONFIG.COLOR_MAP.get(src, "#888888")
        ax.add_patch(plt.Rectangle((LEFT, y_cursor - h), WIDTH, h, facecolor=color, alpha=0.9, edgecolor="white"))
        if h > 0.03:
            ax.text(
                LEFT - 0.02, y_cursor - h / 2,
                f"{src}\n{src_stats[src] * 100:.1f}%",
                ha="right", va="center", fontsize=10, fontweight="bold", color=color
            )
        src_coords[src] = {"top": y_cursor, "h": h}
        y_cursor -= (h + GAP)

    y_cursor = 1.0
    tgt_coords = {}
    valid_h_tgt = 1.0 - (len(tgt_order) - 1) * GAP
    for tgt in tgt_order:
        h = float(tgt_stats[tgt] * valid_h_tgt)
        h = max(h, 0.002)
        color = CONFIG.COLOR_MAP.get(tgt, "#888888")
        ax.add_patch(plt.Rectangle((RIGHT - WIDTH, y_cursor - h), WIDTH, h, facecolor=color, alpha=0.9, edgecolor="white"))
        if h > 0.03:
            label_txt = tgt.replace(" ", "\n").replace(":", "")
            ax.text(
                RIGHT + 0.02, y_cursor - h / 2,
                f"{label_txt}\n{tgt_stats[tgt] * 100:.1f}%",
                ha="left", va="center", fontsize=10, fontweight="bold", color=color
            )
        tgt_coords[tgt] = {"top": y_cursor, "h": h}
        y_cursor -= (h + GAP)

    for src in src_order:
        for tgt in tgt_order:
            row = counts[(counts["Source"] == src) & (counts["Target"] == tgt)]
            if row.empty:
                continue
            count = int(row.iloc[0]["Count"])

            src_count = float(src_stats[src] * total)
            tgt_count = float(tgt_stats[tgt] * total)
            if src_count <= 0 or tgt_count <= 0:
                continue

            h_src = (count / src_count) * src_coords[src]["h"]
            h_tgt = (count / tgt_count) * tgt_coords[tgt]["h"]

            y1 = src_coords[src]["top"]
            y2 = tgt_coords[tgt]["top"]

            src_coords[src]["top"] -= h_src
            tgt_coords[tgt]["top"] -= h_tgt

            x = np.linspace(LEFT + WIDTH, RIGHT - WIDTH, 100)
            sigmoid = 1 / (1 + np.exp(-np.linspace(-6, 6, 100)))

            y_upper = y1 - (y1 - y2) * sigmoid
            y_lower = (y1 - h_src) - ((y1 - h_src) - (y2 - h_tgt)) * sigmoid

            color = CONFIG.COLOR_MAP.get(src, "#888888")
            ax.fill_between(x, y_upper, y_lower, color=color, alpha=0.3, linewidth=0)

    ax.axis("off")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.text(LEFT + WIDTH / 2, -0.05, "Baseline Typology", ha="center", fontweight="bold")
    ax.text(RIGHT - WIDTH / 2, -0.05, "Post-Optimization", ha="center", fontweight="bold")


# -------------------------------
# Main
# -------------------------------

def main():
    np.random.seed(CONFIG.SEED)
    plt.rcParams["font.family"] = CONFIG.FONT_FAMILY
    plt.rcParams["pdf.fonttype"] = 42

    out_dir = Path(CONFIG.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))

    if os.path.exists(CONFIG.WHEAT_OPT_CSV):
        df_wheat = pd.read_csv(CONFIG.WHEAT_OPT_CSV)
        wheat_pred = WheatYieldPredictor(CONFIG.WHEAT_MODEL_DIR)
        y_base_wheat = wheat_pred.predict(df_wheat)
        flow_wheat = identify_structural_evolution(df_wheat, "Wheat", y_base_wheat)

        ax1 = fig.add_subplot(1, 2, 1)
        plot_sankey(ax1, flow_wheat, "(a) Wheat Structural Evolution")

        flow_wheat.to_csv(out_dir / "flow_wheat.csv", index=False, encoding="utf-8")
    else:
        print(f"Warning: Wheat input file not found: {CONFIG.WHEAT_OPT_CSV}")

    if os.path.exists(CONFIG.MAIZE_OPT_CSV):
        df_maize = pd.read_csv(CONFIG.MAIZE_OPT_CSV)
        maize_pred = MaizeYieldPredictor(CONFIG.MAIZE_MODEL_DIR)
        y_base_maize = maize_pred.predict(df_maize)
        flow_maize = identify_structural_evolution(df_maize, "Maize", y_base_maize)

        ax2 = fig.add_subplot(1, 2, 2)
        plot_sankey(ax2, flow_maize, "(b) Maize Structural Evolution")

        flow_maize.to_csv(out_dir / "flow_maize.csv", index=False, encoding="utf-8")
    else:
        print(f"Warning: Maize input file not found: {CONFIG.MAIZE_OPT_CSV}")

    save_file = out_dir / "Fig7_Typology_Evolution_Sankey.pdf"
    plt.tight_layout()
    plt.savefig(save_file, format="pdf", bbox_inches="tight")
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    main()
