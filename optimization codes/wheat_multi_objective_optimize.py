# -*- coding: utf-8 -*-
"""
NSGA-III multi-objective optimization for wheat (three objectives).

Objectives: predicted yield, environmental index, economic index.

Data note: the original survey dataset is not included in this repository.
"""
import pandas as pd
import numpy as np
import os
import joblib
import torch
import rtdl
import torch.multiprocessing as mp
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from tqdm import tqdm
import warnings
import time
import random
import traceback

warnings.filterwarnings("ignore")

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class CONFIG:
    SEED = 123
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    MODEL_SAVE_DIRECTORY = os.environ.get('WHEAT_MODEL_DIR', 'saved_wheat_models_ftt')
    INPUT_EXCEL_PATH = os.environ.get('WHEAT_OPT_INPUT', 'data_demo/wheat.xlsx')
    OPTIMIZATION_RESULTS_DIRECTORY = os.environ.get('WHEAT_OPT_OUTDIR', 'results/wheat_multiobjective')

    DECISION_VARIABLES = [
        'Seed', 'SowDate', 'HarvestDate', 'BasalN', 'BasalP2O5',
        'BasalK2O', 'Irrigation1Amount', 'Irrigation2Amount', 'Irrigation3Amount',
        'TopdressingN', 'TopdressingP2O5', 'TopdressingK2O', 'Pesticide'
    ]
    BOUNDS = np.array([
        (150, 400), (273, 309), (511, 541), (75.6, 303.7485),
        (27, 210.54), (27, 160.875), (0, 1650), (0, 1485), (0, 1485),
        (0, 227.7), (0, 303.6), (0, 104.016), (0.63828, 8.819272)
    ])

    POP_SIZE = 92
    N_GEN = 150

    CEC = {'irrigation_power': 0.92, 'N_fertilizer': 1.53, 'P_fertilizer': 1.14, 'K_fertilizer': 0.66,
           'pesticide': 6.58, 'seed': 1.16}
    NEC = {'irrigation_power': 0.12e-3, 'N_fertilizer': 0.89e-3, 'P_fertilizer': 0.54e-3,
           'K_fertilizer': 0.03e-3, 'pesticide': 5.02e-3, 'seed': 0.24e-3}
    KWH_PER_M3_WATER = 1.0

    PRICES = {'wheat_grain': 0.35, 'N_fertilizer': 0.7, 'P_fertilizer': 0.86, 'K_fertilizer': 0.86, 'pesticide': 24.74,
              'seed': 0.6}
    PRICE_WATER = 0.072
    COST_MACHINERY = 432.14
    COST_LABOR = 39.80


class YieldPredictor:
    def __init__(self, model_dir, device_id=0, device_str=None):
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device(f"cuda:{device_id}")

        self.model = None
        self.scaler_X = None
        self.scaler_y_scale = None
        self.scaler_y_mean = None
        self.feature_order = None
        self._load_all(model_dir)

    def _load_all(self, model_dir):
        try:
            # Prefer repository naming; fall back to legacy names if needed.
            scalerX_candidates = [
                os.path.join(model_dir, 'scaler_X.joblib'),
                os.path.join(model_dir, 'scaler_X_for_ftt.joblib'),
            ]
            scalery_candidates = [
                os.path.join(model_dir, 'scaler_y.joblib'),
                os.path.join(model_dir, 'scaler_y_for_ftt.joblib'),
            ]
            model_candidates = [
                os.path.join(model_dir, 'best_model.pt'),
                os.path.join(model_dir, 'final_best_ftt_model.pt'),
            ]
            scaler_X_path = next((p for p in scalerX_candidates if os.path.exists(p)), None)
            scaler_y_path = next((p for p in scalery_candidates if os.path.exists(p)), None)
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            if scaler_X_path is None or scaler_y_path is None:
                raise FileNotFoundError(f'No scaler files found in {model_dir}. Tried: {scalerX_candidates} and {scalery_candidates}')
            if model_path is None:
                raise FileNotFoundError(f'No FTTransformer weights found in {model_dir}. Tried: {model_candidates}')
            self.scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
            self.scaler_y_scale = torch.from_numpy(scaler_y.scale_).float().to(self.device)
            self.scaler_y_mean = torch.from_numpy(scaler_y.mean_).float().to(self.device)
            self.feature_order = self.scaler_X.feature_names_in_
            n_features = len(self.feature_order)
            self.model = rtdl.FTTransformer.make_default(n_num_features=n_features, cat_cardinalities=None, d_out=1).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            raise e

    @torch.no_grad()
    def predict_batch(self, input_features_df: pd.DataFrame) -> np.ndarray:
        input_features_ordered = input_features_df[self.feature_order]
        input_features_ordered = input_features_ordered.astype(float)
        X_scaled_np = self.scaler_X.transform(input_features_ordered)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32).to(self.device)
        y_pred_scaled = self.model(X_tensor, x_cat=None)
        return (y_pred_scaled * self.scaler_y_scale + self.scaler_y_mean).cpu().numpy().flatten()


class ObjectiveCalculator:
    def __init__(self, predictor, fixed_vars_df, decision_vars_list, baseline_stats):
        self.predictor = predictor
        self.decision_vars_list = decision_vars_list
        self.baseline_stats = baseline_stats
        self.feature_order = predictor.feature_order
        self.decision_vars_map = {name: i for i, name in enumerate(decision_vars_list)}


        if 'GrowingPeriod' not in fixed_vars_df.columns:
            if 'HarvestDate' in fixed_vars_df.columns and 'SowDate' in fixed_vars_df.columns:
                fixed_vars_df['GrowingPeriod'] = fixed_vars_df['HarvestDate'] - fixed_vars_df['SowDate']
            else:
                fixed_vars_df['GrowingPeriod'] = 0

        self.fixed_vars_dict = fixed_vars_df.iloc[0].to_dict()

    def calculate_objectives(self, x: np.ndarray) -> dict:
        pop_size = x.shape[0]

        data_dict = {k: [v] * pop_size for k, v in self.fixed_vars_dict.items()}
        for i, var in enumerate(self.decision_vars_list):
            data_dict[var] = x[:, i]

        current_batch_df = pd.DataFrame(data_dict)
        current_batch_df['GrowingPeriod'] = current_batch_df['HarvestDate'] - current_batch_df['SowDate']

        yield_pred = self.predictor.predict_batch(current_batch_df)

        var_map = self.decision_vars_map
        seed = x[:, var_map['Seed']]
        N = x[:, var_map['BasalN']] + x[:, var_map['TopdressingN']]
        P = x[:, var_map['BasalP2O5']] + x[:, var_map['TopdressingP2O5']]
        K = x[:, var_map['BasalK2O']] + x[:, var_map['TopdressingK2O']]
        TF = N + P + K
        IWR = x[:, var_map['Irrigation1Amount']] + x[:, var_map['Irrigation2Amount']] + x[:,
                                                                                        var_map['Irrigation3Amount']]
        TP = x[:, var_map['Pesticide']]

        eps = 1e-6

        ip = IWR * CONFIG.KWH_PER_M3_WATER

        ce = ip * CONFIG.CEC['irrigation_power'] + N * CONFIG.CEC['N_fertilizer'] + \
             P * CONFIG.CEC['P_fertilizer'] + K * CONFIG.CEC['K_fertilizer'] + \
             TP * CONFIG.CEC['pesticide'] + seed * CONFIG.CEC['seed']

        ne_upstream = ip * CONFIG.NEC['irrigation_power'] + \
                      N * CONFIG.NEC['N_fertilizer'] + P * CONFIG.NEC['P_fertilizer'] + \
                      K * CONFIG.NEC['K_fertilizer'] + TP * CONFIG.NEC['pesticide'] + \
                      seed * CONFIG.NEC['seed']

        n2o_field = N * 0.0105 * (44 / 28) * 0.476
        nh3_field = N * 0.16 * (17 / 14) * 0.833
        no3_field = N * 0.14 * (62 / 14) * 0.238

        ne_field = n2o_field + nh3_field + no3_field

        ne = ne_upstream + ne_field

        CE_int = ce / (yield_pred + eps)
        NE_int = ne / (yield_pred + eps)
        WF_blue = IWR / (yield_pred + eps)

        stats = self.baseline_stats
        z_ce = (stats['CE']['mean'] - CE_int) / stats['CE']['std']
        z_ne = (stats['NE']['mean'] - NE_int) / stats['NE']['std']
        z_wf = (stats['WF']['mean'] - WF_blue) / stats['WF']['std']
        z_tf = (stats['TF']['mean'] - TF) / stats['TF']['std']
        z_iwr = (stats['IWR']['mean'] - IWR) / stats['IWR']['std']
        z_tp = (stats['TP']['mean'] - TP) / stats['TP']['std']

        env_idx = np.mean(np.array([z_ce, z_ne, z_wf, z_tf, z_iwr, z_tp]), axis=0)

        rev = yield_pred * CONFIG.PRICES['wheat_grain']
        cost = (N * CONFIG.PRICES['N_fertilizer'] + P * CONFIG.PRICES['P_fertilizer'] +
                K * CONFIG.PRICES['K_fertilizer'] + TP * CONFIG.PRICES['pesticide'] +
                seed * CONFIG.PRICES['seed']) + \
               (IWR * CONFIG.PRICE_WATER) + (CONFIG.COST_MACHINERY + CONFIG.COST_LABOR)

        prof = rev - cost
        roi = prof / (cost + eps)
        z_prof = (prof - stats['Profit']['mean']) / stats['Profit']['std']
        z_roi = (roi - stats['ROI']['mean']) / stats['ROI']['std']
        econ_idx = np.mean(np.array([z_prof, z_roi]), axis=0)
        return {
            'objectives': np.stack([yield_pred, env_idx, econ_idx], axis=1),
        }


class WheatOptimizationProblem(Problem):
    def __init__(self, n_obj, active_objectives, calculator, xl, xu):
        super().__init__(n_var=len(xl), n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
        self.calculator = calculator
        self.active_objectives = active_objectives

    def _evaluate(self, x, out, *args, **kwargs):
        res = self.calculator.calculate_objectives(x)
        obj_map = {'yield': 0, 'env': 1, 'econ': 2}
        sel_idx = [obj_map[k] for k in self.active_objectives]
        out['F'] = -res['objectives'][:, sel_idx]


def gpu_worker(gpu_id, subset_indices, subset_data, config_dict, baseline_stats, result_queue):
    import warnings
    warnings.filterwarnings("ignore")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device_str = "cuda:0"

    seed_val = config_dict['SEED']
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_val)

    try:
        predictor = YieldPredictor(config_dict['MODEL_SAVE_DIRECTORY'], device_str=device_str)
    except Exception as e:
        print(f"GPU {gpu_id} : {e}")
        result_queue.put([])
        return

    local_results = []
    iterable = tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"GPU {gpu_id}", position=gpu_id, leave=False)

    for idx_in_subset, row in iterable:
        original_idx = subset_indices[idx_in_subset]

        if 'Yield' in row:
            orig_yield = row['Yield']
        elif 'yield' in row:
            orig_yield = row['yield']


        current_fixed_vars_df = pd.DataFrame([row])
        for col in config_dict['DECISION_VARIABLES']:
            if col not in current_fixed_vars_df.columns: current_fixed_vars_df[col] = 0.0
        if 'HarvestDate' not in current_fixed_vars_df: current_fixed_vars_df['HarvestDate'] = 0
        if 'SowDate' not in current_fixed_vars_df: current_fixed_vars_df['SowDate'] = 0

        for col in predictor.feature_order:
            if col in current_fixed_vars_df.columns:
                current_fixed_vars_df[col] = pd.to_numeric(current_fixed_vars_df[col], errors='coerce').fillna(0.0)

        calculator = ObjectiveCalculator(
            predictor, current_fixed_vars_df, config_dict['DECISION_VARIABLES'], baseline_stats
        )

        # Three-objective optimization (yield, env, econ).
        acts = ['yield', 'env', 'econ']
        n_obj = 3
        ref_dirs = get_reference_directions('das-dennis', n_dim=n_obj, n_partitions=6)
        pop_size = int(config_dict.get('POP_SIZE', CONFIG.POP_SIZE))
        pop_size = max(pop_size, len(ref_dirs))

        try:
            prob = WheatOptimizationProblem(n_obj, acts, calculator, config_dict['BOUNDS'][:, 0], config_dict['BOUNDS'][:, 1])
            algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
            term = get_termination('n_gen', config_dict['N_GEN'])
            res = minimize(prob, algo, term, seed=seed_val, verbose=False)
            if res.X is not None:
                final_res = calculator.calculate_objectives(res.X)['objectives']
                for i in range(len(res.X)):
                    row_res = {
                        'Original_Index': original_idx,
                        'Objective_Yield': float(final_res[i][0]),
                        'Objective_Environmental_Index': float(final_res[i][1]),
                        'Objective_Economic_Index': float(final_res[i][2]),
                        'Actual_Yield': float(orig_yield),
                    }
                    for name, val in zip(config_dict['DECISION_VARIABLES'], res.X[i]):
                        row_res[f'Opt_{name}'] = float(val)
                    local_results.append(row_res)
        except Exception:
            print(f"Skipping plot {original_idx} due to error")
            traceback.print_exc()
            continue

    result_queue.put(local_results)


def fill_missing_columns_with_defaults(df):
    for col in CONFIG.DECISION_VARIABLES:
        col_lower = col.lower()
        existing_cols_lower = {c.lower(): c for c in df.columns}
        if col_lower not in existing_cols_lower:
            df[col] = 0.0
        else:
            real_name = existing_cols_lower[col_lower]
            if real_name != col:
                df.rename(columns={real_name: col}, inplace=True)
    if 'yield' in df.columns and 'Yield' not in df.columns:
        df.rename(columns={'yield': 'Yield'}, inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(CONFIG.SEED)
    random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)
    print(f"Wheat multi-objective optimization (three objectives) | Seed={CONFIG.SEED}")

    if CONFIG.NUM_GPUS == 0: exit("GPU")

    df_full = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_full = fill_missing_columns_with_defaults(df_full)

    if 'Opt.Yield' not in df_full.columns:
        df_full['Opt.Yield'] = np.inf
        df_full['Opt.Environmental_Index'] = np.inf
        df_full['Opt.Economic_Index'] = np.inf

    df_std = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_std = fill_missing_columns_with_defaults(df_std)

    # ==========================================================================
    # ==========================================================================
    print("  (Model Baseline)...")
    baseline_predictor = YieldPredictor(CONFIG.MODEL_SAVE_DIRECTORY, device_id=0)

    if 'HarvestDate' in df_std.columns and 'SowDate' in df_std.columns:
        df_std['GrowingPeriod'] = df_std['HarvestDate'] - df_std['SowDate']
    else:
        df_std['GrowingPeriod'] = 0

    for col in baseline_predictor.feature_order:
        if col in df_std.columns:
            df_std[col] = pd.to_numeric(df_std[col], errors='coerce').fillna(0.0)

    baseline_yield_pred = baseline_predictor.predict_batch(df_std)
    print(f"  (Mean: {baseline_yield_pred.mean():.2f})")

    del baseline_predictor
    torch.cuda.empty_cache()

    d = df_std
    yld = baseline_yield_pred

    sw = d['Seed']

    N, P, K = d['BasalN'] + d['TopdressingN'], d['BasalP2O5'] + d['TopdressingP2O5'], d['BasalK2O'] + d[
        'TopdressingK2O']
    IWR, TP, TF = d['Irrigation1Amount'] + d['Irrigation2Amount'] + d['Irrigation3Amount'], d['Pesticide'], N + P + K

    ip = IWR * CONFIG.KWH_PER_M3_WATER

    ce = ip * CONFIG.CEC['irrigation_power'] + N * CONFIG.CEC['N_fertilizer'] + \
         P * CONFIG.CEC['P_fertilizer'] + K * CONFIG.CEC['K_fertilizer'] + \
         TP * CONFIG.CEC['pesticide'] + sw * CONFIG.CEC['seed']

    ne_upstream = ip * CONFIG.NEC['irrigation_power'] + \
                  N * CONFIG.NEC['N_fertilizer'] + P * CONFIG.NEC['P_fertilizer'] + \
                  K * CONFIG.NEC['K_fertilizer'] + TP * CONFIG.NEC['pesticide'] + \
                  sw * CONFIG.NEC['seed']

    n2o = N * 0.0105 * (44 / 28) * 0.476
    nh3 = N * 0.16 * (17 / 14) * 0.833
    no3 = N * 0.14 * (62 / 14) * 0.238
    ne = ne_upstream + n2o + nh3 + no3

    eps = 1e-6
    CI, NI, WF = ce / (yld + eps), ne / (yld + eps), IWR / (yld + eps)

    rev = yld * CONFIG.PRICES['wheat_grain']
    cost = (N * CONFIG.PRICES['N_fertilizer'] + P * CONFIG.PRICES['P_fertilizer'] +
            K * CONFIG.PRICES['K_fertilizer'] + TP * CONFIG.PRICES['pesticide'] +
            sw * CONFIG.PRICES['seed']) + \
           (IWR * CONFIG.PRICE_WATER) + (CONFIG.COST_MACHINERY + CONFIG.COST_LABOR)

    prof = rev - cost
    roi = prof / (cost + eps)

    stats = {
        'CE': {'mean': CI.mean(), 'std': CI.std()},
        'NE': {'mean': NI.mean(), 'std': NI.std()},
        'WF': {'mean': WF.mean(), 'std': WF.std()},
        'TF': {'mean': TF.mean(), 'std': TF.std()},
        'IWR': {'mean': IWR.mean(), 'std': IWR.std()},
        'TP': {'mean': TP.mean(), 'std': TP.std()},
        'Profit': {'mean': prof.mean(), 'std': prof.std()},
        'ROI': {'mean': roi.mean(), 'std': roi.std()}
    }

    config_dict = {k: v for k, v in vars(CONFIG).items() if not k.startswith('__')}
    tasks = [(index, row) for index, row in df_full.iterrows()]

    print(f"\nProcessing {len(tasks)} ...")

    chunks = np.array_split(df_full, CONFIG.NUM_GPUS)
    procs = []
    q = mp.Queue()

    for i in range(CONFIG.NUM_GPUS):
        idx = chunks[i].index.tolist()
        dat = chunks[i].reset_index(drop=True)
        p = mp.Process(target=gpu_worker, args=(i, idx, dat, config_dict, stats, q))
        p.start()
        procs.append(p)

    all_res = []
    active = len(procs)
    while active > 0:
        if not q.empty():
            all_res.extend(q.get())
            active -= 1
        time.sleep(0.5)

    for p in procs: p.join()

    if all_res:
        df_all = pd.DataFrame(all_res)


        os.makedirs(CONFIG.OPTIMIZATION_RESULTS_DIRECTORY, exist_ok=True)
        merged = pd.merge(
            df_full.reset_index().rename(columns={'index': 'Original_Index'}),
            df_all,
            on='Original_Index',
            how='right',
        )
        out_path = f"{CONFIG.OPTIMIZATION_RESULTS_DIRECTORY}/optimized_wheat_multiobjective.csv"
        merged.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"Saved: {out_path}")
