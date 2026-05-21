# -*- coding: utf-8 -*-
"""
NSGA-III multi-objective optimization for maize (three objectives).

Objectives: predicted yield, environmental index, economic index.

Data note: the original survey dataset is not included in this repository.
Provide your own input file with the same schema.
"""

import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CONFIG:
    SEED = 123
    CPU_CORES = int(os.environ.get("MAIZE_CPU_CORES", "-1"))

    MODEL_SAVE_DIRECTORY = os.environ.get("MAIZE_MODEL_DIR", "models/saved_maize_models_xgb")
    INPUT_EXCEL_PATH = os.environ.get("MAIZE_OPT_INPUT", "data_demo/maize.xlsx")
    OPTIMIZATION_RESULTS_DIRECTORY = os.environ.get("MAIZE_OPT_OUTDIR", "results/maize_multiobjective")

    DECISION_VARIABLES = [
        "Density",
        "SowDate",
        "HarvestDate",
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
    ]

    BOUNDS = np.array(
        [
            (57428, 114188),
            (148, 173),
            (265, 290),
            (121.5, 363.825),
            (31.6928571428571, 163.24),
            (31.725, 171.424),
            (121.5, 990),
            (0, 577.5),
            (0, 495),
            (0, 151.8),
            (0, 0),
            (0, 72.1875),
            (0.1809, 2.1755525),
        ],
        dtype=float,
    )

    POP_SIZE = int(os.environ.get("MAIZE_NSGA_POP_SIZE", "92"))
    N_GEN = int(os.environ.get("MAIZE_NSGA_N_GEN", "150"))

    CEC = {
        "irrigation_power": 0.92,
        "N_fertilizer": 1.53,
        "P_fertilizer": 1.14,
        "K_fertilizer": 0.66,
        "pesticide": 6.58,
        "seed": 1.22,
    }
    NEC = {
        "irrigation_power": 0.12e-3,
        "N_fertilizer": 0.89e-3,
        "P_fertilizer": 0.54e-3,
        "K_fertilizer": 0.03e-3,
        "pesticide": 5.02e-3,
        "seed": 0.88e-3,
    }

    KWH_PER_M3_WATER = 1.0
    AVG_SEED_WEIGHT_GRAM = 344.0

    PRICES = {
        "maize_grain": 0.35,
        "N_fertilizer": 0.7,
        "P_fertilizer": 0.86,
        "K_fertilizer": 0.86,
        "pesticide": 24.74,
        "seed": 5.57,
    }
    PRICE_WATER = 0.072
    COST_MACHINERY = 151.06
    COST_LABOR = 73.09


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    existing_lower = {c.lower(): c for c in df.columns}
    for c in CONFIG.DECISION_VARIABLES:
        c_lower = c.lower()
        if c_lower not in existing_lower:
            df[c] = 0.0
        else:
            real = existing_lower[c_lower]
            if real != c:
                df.rename(columns={real: c}, inplace=True)
    if "yield" in df.columns and "Yield" not in df.columns:
        df.rename(columns={"yield": "Yield"}, inplace=True)
    return df


class YieldPredictor:
    def __init__(self, model_dir: str):
        self.model = None
        self.scaler_X = None
        self.feature_order = None
        self._load_all(model_dir)

    def _load_all(self, model_dir: str) -> None:
        scaler_candidates = [
            os.path.join(model_dir, "scaler_X.joblib"),
            os.path.join(model_dir, "scaler_for_xgb.joblib"),
        ]
        model_candidates = [
            os.path.join(model_dir, "xgb_model.json"),
            os.path.join(model_dir, "manual_best_xgb_model.json"),
        ]
        scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
        model_path = next((p for p in model_candidates if os.path.exists(p)), None)
        if scaler_path is None:
            raise FileNotFoundError(f"No X scaler found in {model_dir}. Tried: {scaler_candidates}")
        if model_path is None:
            raise FileNotFoundError(f"No XGBoost model found in {model_dir}. Tried: {model_candidates}")

        self.scaler_X = joblib.load(scaler_path)
        self.feature_order = list(self.scaler_X.feature_names_in_)

        self.model = xgb.XGBRegressor(tree_method="hist", n_jobs=1)
        self.model.load_model(model_path)

    def predict_batch(self, input_features_df: pd.DataFrame) -> np.ndarray:
        input_features_ordered = input_features_df[self.feature_order].astype(float)
        X_scaled = self.scaler_X.transform(input_features_ordered)
        return self.model.predict(X_scaled)


class ObjectiveCalculator:
    def __init__(self, predictor: YieldPredictor, fixed_vars_df: pd.DataFrame, decision_vars_list, baseline_stats):
        self.predictor = predictor
        self.decision_vars_list = decision_vars_list
        self.baseline_stats = baseline_stats
        self.feature_order = predictor.feature_order
        self.decision_vars_map = {name: i for i, name in enumerate(decision_vars_list)}

        fixed_vars_df = fixed_vars_df.copy()
        if "GrowingPeriod" not in fixed_vars_df.columns:
            if "HarvestDate" in fixed_vars_df.columns and "SowDate" in fixed_vars_df.columns:
                fixed_vars_df["GrowingPeriod"] = fixed_vars_df["HarvestDate"] - fixed_vars_df["SowDate"]
            else:
                fixed_vars_df["GrowingPeriod"] = 0

        self.fixed_vars_dict = fixed_vars_df.iloc[0].to_dict()

    def calculate_objectives(self, x: np.ndarray) -> dict:
        pop_size = x.shape[0]

        data_dict = {k: [v] * pop_size for k, v in self.fixed_vars_dict.items()}
        for i, var in enumerate(self.decision_vars_list):
            data_dict[var] = x[:, i]

        batch_df = pd.DataFrame(data_dict)
        batch_df["GrowingPeriod"] = batch_df["HarvestDate"] - batch_df["SowDate"]

        yld_pred = self.predictor.predict_batch(batch_df)

        v = self.decision_vars_map
        den = x[:, v["Density"]]
        sw = den * CONFIG.AVG_SEED_WEIGHT_GRAM / 1000000.0
        N = x[:, v["BasalN"]] + x[:, v["TopdressingN"]]
        P = x[:, v["BasalP2O5"]] + x[:, v["TopdressingP2O5"]]
        K = x[:, v["BasalK2O"]] + x[:, v["TopdressingK2O"]]
        TF = N + P + K
        IWR = x[:, v["Irrigation1Amount"]] + x[:, v["Irrigation2Amount"]] + x[:, v["Irrigation3Amount"]]
        TP = x[:, v["Pesticide"]]

        eps = 1e-6

        ip = IWR * CONFIG.KWH_PER_M3_WATER
        ce = (
            ip * CONFIG.CEC["irrigation_power"]
            + N * CONFIG.CEC["N_fertilizer"]
            + P * CONFIG.CEC["P_fertilizer"]
            + K * CONFIG.CEC["K_fertilizer"]
            + TP * CONFIG.CEC["pesticide"]
            + sw * CONFIG.CEC["seed"]
        )
        ne_up = (
            ip * CONFIG.NEC["irrigation_power"]
            + N * CONFIG.NEC["N_fertilizer"]
            + P * CONFIG.NEC["P_fertilizer"]
            + K * CONFIG.NEC["K_fertilizer"]
            + TP * CONFIG.NEC["pesticide"]
            + sw * CONFIG.NEC["seed"]
        )
        n2o = N * 0.0250 * (44 / 28) * 0.476
        nh3 = N * 0.21 * (17 / 14) * 0.833
        no3 = N * 0.19 * (62 / 14) * 0.238
        ne = ne_up + n2o + nh3 + no3

        CI = ce / (yld_pred + eps)
        NI = ne / (yld_pred + eps)
        WF = IWR / (yld_pred + eps)

        st = self.baseline_stats
        z_ci = (st["CE"]["mean"] - CI) / st["CE"]["std"]
        z_ni = (st["NE"]["mean"] - NI) / st["NE"]["std"]
        z_wf = (st["WF"]["mean"] - WF) / st["WF"]["std"]
        z_tf = (st["TF"]["mean"] - TF) / st["TF"]["std"]
        z_iwr = (st["IWR"]["mean"] - IWR) / st["IWR"]["std"]
        z_tp = (st["TP"]["mean"] - TP) / st["TP"]["std"]
        env_idx = np.mean(np.array([z_ci, z_ni, z_wf, z_tf, z_iwr, z_tp]), axis=0)

        rev = yld_pred * CONFIG.PRICES["maize_grain"]
        cost = (
            N * CONFIG.PRICES["N_fertilizer"]
            + P * CONFIG.PRICES["P_fertilizer"]
            + K * CONFIG.PRICES["K_fertilizer"]
            + TP * CONFIG.PRICES["pesticide"]
            + sw * CONFIG.PRICES["seed"]
            + IWR * CONFIG.PRICE_WATER
            + CONFIG.COST_MACHINERY
            + CONFIG.COST_LABOR
        )
        prof = rev - cost
        roi = prof / (cost + eps)
        z_prof = (prof - st["Profit"]["mean"]) / st["Profit"]["std"]
        z_roi = (roi - st["ROI"]["mean"]) / st["ROI"]["std"]
        econ_idx = np.mean(np.array([z_prof, z_roi]), axis=0)

        return {"objectives": np.stack([yld_pred, env_idx, econ_idx], axis=1)}


class CornOptimizationProblem(Problem):
    def __init__(self, calculator: ObjectiveCalculator, xl: np.ndarray, xu: np.ndarray):
        super().__init__(n_var=len(xl), n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.calculator = calculator

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.calculator.calculate_objectives(x)
        out["F"] = -res["objectives"]


def choose_representative_solution(final_obj: np.ndarray) -> int:
    if final_obj.shape[0] <= 1:
        return 0
    norm = (final_obj - final_obj.min(0)) / (final_obj.max(0) - final_obj.min(0) + 1e-9)
    return int(np.argmin(np.linalg.norm(norm - 1.0, axis=1)))


def process_single_plot(task_tuple, predictor: YieldPredictor, config_dict, baseline_stats):
    seed_val = config_dict["SEED"]
    np.random.seed(seed_val)
    random.seed(seed_val)

    idx, row = task_tuple

    curr_fixed = pd.DataFrame([row])
    curr_fixed = fill_missing(curr_fixed)

    calculator = ObjectiveCalculator(predictor, curr_fixed, config_dict["DECISION_VARIABLES"], baseline_stats)

    acts = ["yield", "env", "econ"]
    n_obj = 3
    ref_dirs = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=6)

    pop_size = int(config_dict.get("POP_SIZE", CONFIG.POP_SIZE))
    pop_size = max(pop_size, len(ref_dirs))

    try:
        prob = CornOptimizationProblem(calculator, config_dict["BOUNDS"][:, 0], config_dict["BOUNDS"][:, 1])
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
        term = get_termination("n_gen", config_dict["N_GEN"])
        res = minimize(prob, algo, term, seed=config_dict["SEED"], verbose=False)
        if res.X is None:
            return []

        final = calculator.calculate_objectives(res.X)["objectives"]
        best_idx = choose_representative_solution(final)

        r = {
            "Original_Index": int(idx),
            "Objective_Yield": float(final[best_idx][0]),
            "Objective_Environmental_Index": float(final[best_idx][1]),
            "Objective_Economic_Index": float(final[best_idx][2]),
        }
        for k, v in zip(config_dict["DECISION_VARIABLES"], res.X[best_idx]):
            r[f"Opt_{k}"] = float(v)
        return [r]
    except Exception:
        return []


if __name__ == "__main__":
    np.random.seed(CONFIG.SEED)
    random.seed(CONFIG.SEED)
    print(f"Maize multi-objective optimization (three objectives) | Seed={CONFIG.SEED}")

    df_full = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_full = fill_missing(df_full)

    predictor = YieldPredictor(CONFIG.MODEL_SAVE_DIRECTORY)

    if "GrowingPeriod" not in df_full.columns:
        if "HarvestDate" in df_full.columns and "SowDate" in df_full.columns:
            df_full["GrowingPeriod"] = df_full["HarvestDate"] - df_full["SowDate"]
        else:
            df_full["GrowingPeriod"] = 0

    for col in predictor.feature_order:
        if col in df_full.columns:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce").fillna(0.0)

    baseline_yld = predictor.predict_batch(df_full)
    d = df_full
    yld = baseline_yld

    den = d["Density"]
    sw = den * CONFIG.AVG_SEED_WEIGHT_GRAM / 1000000.0
    N = d["BasalN"] + d["TopdressingN"]
    P = d["BasalP2O5"] + d["TopdressingP2O5"]
    K = d["BasalK2O"] + d["TopdressingK2O"]
    TF = N + P + K
    IWR = d["Irrigation1Amount"] + d["Irrigation2Amount"] + d["Irrigation3Amount"]
    TP = d["Pesticide"]

    ip = IWR * CONFIG.KWH_PER_M3_WATER
    ce = ip * 0.92 + N * 1.53 + P * 1.14 + K * 0.66 + TP * 6.58 + sw * 1.22
    ne_up = ip * 0.12e-3 + N * 0.89e-3 + P * 0.54e-3 + K * 0.03e-3 + TP * 5.02e-3 + sw * 0.88e-3
    n2o = N * 0.0250 * (44 / 28) * 0.476
    nh3 = N * 0.21 * (17 / 14) * 0.833
    no3 = N * 0.19 * (62 / 14) * 0.238
    ne = ne_up + n2o + nh3 + no3

    eps = 1e-6
    CI = ce / (yld + eps)
    NI = ne / (yld + eps)
    WF = IWR / (yld + eps)

    rev = yld * CONFIG.PRICES["maize_grain"]
    cost = (N * 0.7 + P * 0.86 + K * 0.86 + TP * 24.74 + sw * 5.57) + IWR * CONFIG.PRICE_WATER + CONFIG.COST_MACHINERY + CONFIG.COST_LABOR
    prof = rev - cost
    roi = prof / (cost + eps)

    def ms(a: np.ndarray) -> dict:
        s = float(np.std(a))
        if not np.isfinite(s) or s < 1e-12:
            s = 1e-6
        return {"mean": float(np.mean(a)), "std": s}

    stats = {
        "CE": ms(CI.to_numpy() if hasattr(CI, "to_numpy") else np.asarray(CI)),
        "NE": ms(NI.to_numpy() if hasattr(NI, "to_numpy") else np.asarray(NI)),
        "WF": ms(WF.to_numpy() if hasattr(WF, "to_numpy") else np.asarray(WF)),
        "TF": ms(TF.to_numpy() if hasattr(TF, "to_numpy") else np.asarray(TF)),
        "IWR": ms(IWR.to_numpy() if hasattr(IWR, "to_numpy") else np.asarray(IWR)),
        "TP": ms(TP.to_numpy() if hasattr(TP, "to_numpy") else np.asarray(TP)),
        "Profit": ms(prof.to_numpy() if hasattr(prof, "to_numpy") else np.asarray(prof)),
        "ROI": ms(roi.to_numpy() if hasattr(roi, "to_numpy") else np.asarray(roi)),
    }

    config_dict = {k: v for k, v in vars(CONFIG).items() if not k.startswith("__")}
    tasks = [(index, row) for index, row in df_full.iterrows()]

    res_nested = Parallel(n_jobs=CONFIG.CPU_CORES)(
        delayed(process_single_plot)(t, predictor, config_dict, stats) for t in tqdm(tasks, desc="Optimizing plots")
    )
    all_res = [x for sub in res_nested for x in sub]

    if all_res:
        df_all = pd.DataFrame(all_res)
        os.makedirs(CONFIG.OPTIMIZATION_RESULTS_DIRECTORY, exist_ok=True)
        merged = pd.merge(
            df_full.reset_index().rename(columns={"index": "Original_Index"}),
            df_all,
            on="Original_Index",
            how="right",
        )
        out_path = f"{CONFIG.OPTIMIZATION_RESULTS_DIRECTORY}/optimized_maize_multiobjective.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {out_path}")
