# -*- coding: utf-8 -*-
"""
Single-objective optimization for wheat using an FTTransformer surrogate model.
"""
import pandas as pd
import numpy as np
import os
import joblib
import torch
import rtdl
import torch.multiprocessing as mp
from scipy.optimize import differential_evolution
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

    OUTPUT_DIR = os.environ.get('WHEAT_OPT_OUTPUT_DIR', 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATHS = {
        'Yield': os.path.join(OUTPUT_DIR, 'wheat_single_objective_yield.xlsx'),
        'Env': os.path.join(OUTPUT_DIR, 'wheat_single_objective_environment.xlsx'),
        'Econ': os.path.join(OUTPUT_DIR, 'wheat_single_objective_economic.xlsx'),
    }

    DECISION_VARIABLES = [
        'Seed', 'SowDate', 'HarvestDate', 'BasalN', 'BasalP2O5',
        'BasalK2O', 'Irrigation1Amount', 'Irrigation2Amount', 'Irrigation3Amount',
        'TopdressingN', 'TopdressingP2O5', 'TopdressingK2O', 'Pesticide'
    ]

    VAR_MAPPING = {
        'Seed': 'Opt.Seed', 'SowDate': 'Opt.SowDate', 'HarvestDate': 'Opt.HarvestDate',
        'BasalN': 'Opt.BasalN', 'BasalP2O5': 'Opt.BasalP2O5', 'BasalK2O': 'Opt.BasalK2O',
        'Irrigation1Amount': 'Opt.Irrigation1Amount', 'Irrigation2Amount': 'Opt.Irrigation2Amount',
        'Irrigation3Amount': 'Opt.Irrigation3Amount', 'TopdressingN': 'Opt.TopdressingN',
        'TopdressingP2O5': 'Opt.TopdressingP2O5', 'TopdressingK2O': 'Opt.TopdressingK2O',
        'Pesticide': 'Opt.Pesticide'
    }

    BOUNDS = [
        (150, 400), (273, 309), (511, 541), (75.6, 303.7485),
        (27, 210.54), (27, 160.875), (0, 1650), (0, 1485), (0, 1485),
        (0, 227.7), (0, 303.6), (0, 104.016), (0.63828, 8.819272)
    ]

    DE_STRATEGY = 'rand1bin'
    DE_POPSIZE = 200
    DE_MAXITER = 500

    TOL_MAPPING = {'Yield': 0.001, 'Env': 0.01, 'Econ': 0.01}

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
    def __init__(self, model_dir, device_str):
        self.device = torch.device(device_str)
        self.model = None
        self.scaler_X = None
        self.scaler_y_scale = None
        self.scaler_y_mean = None
        self.feature_order = None
        self._load_all(model_dir)

    def _load_all(self, model_dir):
        try:
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
        X_scaled_np = self.scaler_X.transform(input_features_ordered)
        X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32).to(self.device)
        y_pred_scaled = self.model(X_tensor, x_cat=None)
        return (y_pred_scaled * self.scaler_y_scale + self.scaler_y_mean).cpu().numpy().flatten()


def calculate_metrics_batch(decision_vars_values, predictor, fixed_vars_df, decision_vars_list, baseline_stats):
    is_population = decision_vars_values.ndim == 2
    if is_population:
        decision_vars_df = pd.DataFrame(decision_vars_values.T, columns=decision_vars_list)
        pop_size = decision_vars_df.shape[0]
        current_batch_df = pd.concat([fixed_vars_df] * pop_size, ignore_index=True)
        for var_name in decision_vars_list:
            current_batch_df[var_name] = decision_vars_df[var_name]
        x_arr = decision_vars_values.T
    else:
        decision_vars_df = pd.DataFrame(decision_vars_values.reshape(1, -1), columns=decision_vars_list)
        current_batch_df = fixed_vars_df.copy()
        for var_name in decision_vars_list:
            current_batch_df[var_name] = decision_vars_df[var_name]
        x_arr = decision_vars_values.reshape(1, -1)

    current_batch_df['GrowingPeriod'] = current_batch_df['HarvestDate'] - current_batch_df['SowDate']
    yield_pred = predictor.predict_batch(current_batch_df)

    var_map = {name: i for i, name in enumerate(decision_vars_list)}
    seed = x_arr[:, var_map['Seed']]
    N = x_arr[:, var_map['BasalN']] + x_arr[:, var_map['TopdressingN']]
    P = x_arr[:, var_map['BasalP2O5']] + x_arr[:, var_map['TopdressingP2O5']]
    K = x_arr[:, var_map['BasalK2O']] + x_arr[:, var_map['TopdressingK2O']]
    TF = N + P + K
    IWR = x_arr[:, var_map['Irrigation1Amount']] + x_arr[:, var_map['Irrigation2Amount']] + x_arr[:, var_map[
                                                                                                         'Irrigation3Amount']]
    TP = x_arr[:, var_map['Pesticide']]
    eps = 1e-6

    ip = IWR * CONFIG.KWH_PER_M3_WATER
    ce = ip * CONFIG.CEC['irrigation_power'] + N * CONFIG.CEC['N_fertilizer'] + \
         P * CONFIG.CEC['P_fertilizer'] + K * CONFIG.CEC['K_fertilizer'] + \
         TP * CONFIG.CEC['pesticide'] + seed * CONFIG.CEC['seed']

    ne_upstream = ip * CONFIG.NEC['irrigation_power'] + \
                  N * CONFIG.NEC['N_fertilizer'] + P * CONFIG.NEC['P_fertilizer'] + \
                  K * CONFIG.NEC['K_fertilizer'] + TP * CONFIG.NEC['pesticide'] + \
                  seed * CONFIG.NEC['seed']
    n2o = N * 0.0105 * (44 / 28) * 0.476
    nh3 = N * 0.16 * (17 / 14) * 0.833
    no3 = N * 0.14 * (62 / 14) * 0.238
    ne = ne_upstream + n2o + nh3 + no3

    CI = ce / (yield_pred + eps)
    NI = ne / (yield_pred + eps)
    WF = IWR / (yield_pred + eps)

    stats = baseline_stats
    z_ci = (stats['CI']['mean'] - CI) / stats['CI']['std']
    z_ni = (stats['NI']['mean'] - NI) / stats['NI']['std']
    z_wf = (stats['WF']['mean'] - WF) / stats['WF']['std']
    z_tf = (stats['TF']['mean'] - TF) / stats['TF']['std']
    z_iwr = (stats['IWR']['mean'] - IWR) / stats['IWR']['std']
    z_tp = (stats['TP']['mean'] - TP) / stats['TP']['std']
    environmental_index = np.mean(np.array([z_ci, z_ni, z_wf, z_tf, z_iwr, z_tp]), axis=0)

    prod_val = yield_pred * CONFIG.PRICES['wheat_grain']
    cost = (N * 0.7 + P * 0.86 + K * 0.86 + TP * 24.74 + seed * 0.6) + (IWR * CONFIG.PRICE_WATER) + (
            CONFIG.COST_MACHINERY + CONFIG.COST_LABOR)
    prof = prod_val - cost
    roi = prof / (cost + eps)

    z_prof = (prof - stats['Profit']['mean']) / stats['Profit']['std']
    z_roi = (roi - stats['ROI']['mean']) / stats['ROI']['std']
    economic_index = np.mean(np.array([z_prof, z_roi]), axis=0)

    return {
        'Yield': yield_pred,
        'Environmental_Index': environmental_index,
        'Economic_Index': economic_index,
        'GrowingPeriod': current_batch_df['GrowingPeriod'].values
    }


def objective_function_wrapper(decision_vars_values, predictor, fixed_vars_df, decision_vars_list, baseline_stats,
                               mode):
    metrics = calculate_metrics_batch(decision_vars_values, predictor, fixed_vars_df, decision_vars_list,
                                      baseline_stats)
    if mode == 'Yield':
        return -metrics['Yield']
    elif mode == 'Env':
        return -metrics['Environmental_Index']
    elif mode == 'Econ':
        return -metrics['Economic_Index']
    else:
        raise ValueError("Unknown Mode")


def gpu_worker(gpu_id, subset_indices, subset_data, config_dict, baseline_stats, mode, result_queue):
    import warnings
    warnings.filterwarnings("ignore")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device_str = "cuda:0"
    torch.set_num_threads(1)

    seed_val = config_dict['SEED']
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    try:
        predictor = YieldPredictor(config_dict['MODEL_SAVE_DIRECTORY'], device_str=device_str)
    except Exception as e:
        print(f"Error: GPU-{gpu_id} fail: {e}")
        result_queue.put([])
        return

    local_results = []
    current_tol = config_dict['TOL_MAPPING'][mode]
    iterator = tqdm(subset_data.iterrows(), total=len(subset_data), desc=f"GPU-{gpu_id} [{mode}]", position=gpu_id,
                    leave=False)

    for idx_in_subset, row in iterator:
        original_idx = subset_indices[idx_in_subset]
        try:
            current_fixed_vars_df = pd.DataFrame([row])
            for col in config_dict['DECISION_VARIABLES']:
                if col not in current_fixed_vars_df.columns: current_fixed_vars_df[col] = 0.0
            if 'HarvestDate' not in current_fixed_vars_df: current_fixed_vars_df['HarvestDate'] = 0
            if 'SowDate' not in current_fixed_vars_df: current_fixed_vars_df['SowDate'] = 0

            for col in predictor.feature_order:
                if col in current_fixed_vars_df.columns:
                    current_fixed_vars_df[col] = pd.to_numeric(current_fixed_vars_df[col], errors='coerce').fillna(0.0)

            result = differential_evolution(
                func=objective_function_wrapper,
                bounds=config_dict['BOUNDS'],
                args=(predictor, current_fixed_vars_df, config_dict['DECISION_VARIABLES'], baseline_stats, mode),
                strategy=config_dict['DE_STRATEGY'], maxiter=config_dict['DE_MAXITER'],
                popsize=config_dict['DE_POPSIZE'],
                tol=current_tol, vectorized=True, polish=False, disp=False, seed=seed_val
            )
            final_metrics = calculate_metrics_batch(result.x, predictor, current_fixed_vars_df,
                                                    config_dict['DECISION_VARIABLES'], baseline_stats)
            best_params = dict(zip(config_dict['DECISION_VARIABLES'], result.x))

            row_res = {
                'original_index': original_idx,
                'Opt.Yield': final_metrics['Yield'][0],
                'Opt.Environmental_Index': final_metrics['Environmental_Index'][0],
                'Opt.Economic_Index': final_metrics['Economic_Index'][0]
            }
            for k, v in best_params.items():
                row_res[CONFIG.VAR_MAPPING[k]] = v if k not in ['Seed', 'SowDate', 'HarvestDate'] else int(np.round(v))
            local_results.append(row_res)
        except Exception:
            traceback.print_exc()
            continue

    result_queue.put(local_results)


def fill_missing_columns_with_defaults(df):
    for col in CONFIG.DECISION_VARIABLES:
        if col not in df.columns: df[col] = 0.0
    if 'yield' in df.columns and 'Yield' not in df.columns: df.rename(columns={'yield': 'Yield'}, inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(CONFIG.SEED)
    random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)
    print(f"Wheat single-objective optimization | Seed={CONFIG.SEED}")

    if CONFIG.NUM_GPUS == 0: raise RuntimeError('No CUDA device detected. Set CUDA properly or run on a machine with a GPU.')

    df_full = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_full = fill_missing_columns_with_defaults(df_full)
    df_std = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_std = fill_missing_columns_with_defaults(df_std)

    print("Computing baseline predictions...")
    baseline_predictor = YieldPredictor(CONFIG.MODEL_SAVE_DIRECTORY, device_str="cuda:0")
    if 'GrowingPeriod' not in df_std.columns: df_std['GrowingPeriod'] = df_std['HarvestDate'] - df_std['SowDate']
    baseline_yield_pred = baseline_predictor.predict_batch(df_std)
    print(f"Baseline predictions completed (mean={baseline_yield_pred.mean():.2f}).")

    del baseline_predictor
    torch.cuda.empty_cache()

    d = df_std
    yld = baseline_yield_pred
    seed = d['Seed']
    N, P, K = d['BasalN'] + d['TopdressingN'], d['BasalP2O5'] + d['TopdressingP2O5'], d['BasalK2O'] + d[
        'TopdressingK2O']
    TF, IWR, TP = N + P + K, d['Irrigation1Amount'] + d['Irrigation2Amount'] + d['Irrigation3Amount'], d['Pesticide']

    ip = IWR * CONFIG.KWH_PER_M3_WATER
    ce = ip * 0.92 + N * 1.53 + P * 1.14 + K * 0.66 + TP * 6.58 + seed * 1.16
    ne_up = ip * 0.12e-3 + N * 0.89e-3 + P * 0.54e-3 + K * 0.03e-3 + TP * 5.02e-3 + seed * 0.24e-3
    ne_field = N * 0.0105 * (44 / 28) * 0.476  + N * 0.16 * (17 / 14) * 0.833  + N * 0.14 * (
            62 / 14) * 0.238 
    ne = ne_up + ne_field

    eps = 1e-6
    CI, NI, WF = ce / (yld + eps), ne / (yld + eps), IWR / (yld + eps)
    rev = yld * CONFIG.PRICES['wheat_grain']
    cost = (N * 0.7 + P * 0.86 + K * 0.86 + TP * 24.74 + seed * 0.6) + (IWR * CONFIG.PRICE_WATER) + (
            CONFIG.COST_MACHINERY + CONFIG.COST_LABOR)
    prof, roi = rev - cost, (rev - cost) / (cost + eps)

    stats = {
        'CI': {'mean': CI.mean(), 'std': CI.std()}, 'NI': {'mean': NI.mean(), 'std': NI.std()},
        'WF': {'mean': WF.mean(), 'std': WF.std()}, 'TF': {'mean': TF.mean(), 'std': TF.std()},
        'IWR': {'mean': IWR.mean(), 'std': IWR.std()}, 'TP': {'mean': TP.mean(), 'std': TP.std()},
        'Profit': {'mean': prof.mean(), 'std': prof.std()}, 'ROI': {'mean': roi.mean(), 'std': roi.std()}
    }

    config_dict = {k: v for k, v in vars(CONFIG).items() if not k.startswith('__')}

    chunks = np.array_split(df_full, CONFIG.NUM_GPUS)

    for mode in ['Yield', 'Env', 'Econ']:
        print(f"\n : {mode}")
        start = time.time()

        procs = []
        q = mp.Queue()

        for i in range(CONFIG.NUM_GPUS):
            idx = chunks[i].index.tolist()
            dat = chunks[i].reset_index(drop=True)
            p = mp.Process(target=gpu_worker, args=(i, idx, dat, config_dict, stats, mode, q))
            p.start()
            procs.append(p)

        all_res = []
        active_procs = len(procs)

        while active_procs > 0:
            if not q.empty():
                res_list = q.get()
                all_res.extend(res_list)
                active_procs -= 1
            time.sleep(0.5)

        for p in procs:
            p.join()

        if all_res:
            res_df = pd.DataFrame(all_res).set_index('original_index')
            out_df = df_full.copy()
            for c in res_df.columns:
                out_df.loc[res_df.index, c] = res_df[c]

            out_df.to_excel(CONFIG.OUTPUT_PATHS[mode], index=False)
            print(f"Saved: {CONFIG.OUTPUT_PATHS[mode]}")

        print(f"⏱️ : {(time.time() - start) / 60:.2f} min")
