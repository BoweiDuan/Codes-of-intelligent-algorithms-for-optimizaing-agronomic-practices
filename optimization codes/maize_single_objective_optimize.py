# -*- coding: utf-8 -*-
"""
Single-objective optimization for maize using an XGBoost surrogate model.
"""
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import time
import multiprocessing
import random

warnings.filterwarnings("ignore")


class CONFIG:
    N_CORES = int(os.environ.get('MAIZE_N_CORES', str(min(multiprocessing.cpu_count(), 4))))
    SEED = 123
    MODEL_SAVE_DIRECTORY = os.environ.get('MAIZE_MODEL_DIR', 'models/saved_maize_models_xgb')
    INPUT_EXCEL_PATH = os.environ.get('MAIZE_OPT_INPUT', 'data_demo/maize.xlsx')

    OUTPUT_DIR = os.environ.get('MAIZE_OPT_OUTPUT_DIR', 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATHS = {
        'Yield': os.path.join(OUTPUT_DIR, 'maize_single_objective_yield.xlsx'),
        'Env': os.path.join(OUTPUT_DIR, 'maize_single_objective_environment.xlsx'),
        'Econ': os.path.join(OUTPUT_DIR, 'maize_single_objective_economic.xlsx'),
    }

    DECISION_VARIABLES = [
        'Density', 'SowDate', 'HarvestDate', 'BasalN', 'BasalP2O5',
        'BasalK2O', 'Irrigation1Amount', 'Irrigation2Amount', 'Irrigation3Amount',
        'TopdressingN', 'TopdressingP2O5', 'TopdressingK2O', 'Pesticide'
    ]

    VAR_MAPPING = {
        'Density': 'Opt.Density', 'SowDate': 'Opt.SowDate', 'HarvestDate': 'Opt.HarvestDate',
        'BasalN': 'Opt.BasalN', 'BasalP2O5': 'Opt.BasalP2O5', 'BasalK2O': 'Opt.BasalK2O',
        'Irrigation1Amount': 'Opt.Irrigation1Amount', 'Irrigation2Amount': 'Opt.Irrigation2Amount',
        'Irrigation3Amount': 'Opt.Irrigation3Amount', 'TopdressingN': 'Opt.TopdressingN',
        'TopdressingP2O5': 'Opt.TopdressingP2O5', 'TopdressingK2O': 'Opt.TopdressingK2O',
        'Pesticide': 'Opt.Pesticide'
    }

    BOUNDS = np.array([
        (57428, 114188), (148, 173), (265, 290), (121.5, 363.825),
        (31.6928571428571, 163.24), (31.725, 171.424), (121.5, 990),
        (0, 577.5), (0, 495), (0, 151.8), (0, 0), (0, 72.1875), (0.1809, 2.1755525)
    ])

    DE_STRATEGY = 'rand1bin'
    DE_POPSIZE = int(os.environ.get('MAIZE_DE_POPSIZE', '200'))
    DE_MAXITER = int(os.environ.get('MAIZE_DE_MAXITER', '500'))
    TOL_MAPPING = {'Yield': 0.001, 'Env': 0.01, 'Econ': 0.01}

    CEC = {'irrigation_power': 0.92, 'N_fertilizer': 1.53, 'P_fertilizer': 1.14, 'K_fertilizer': 0.66,
           'pesticide': 6.58, 'seed': 1.22}
    NEC = {'irrigation_power': 0.12e-3, 'N_fertilizer': 0.89e-3, 'P_fertilizer': 0.54e-3,
           'K_fertilizer': 0.03e-3, 'pesticide': 5.02e-3, 'seed': 0.88e-3}
    KWH_PER_M3_WATER = 1.0

    AVG_SEED_WEIGHT_GRAM = 344.0

    PRICES = {'maize_grain': 0.35, 'N_fertilizer': 0.7, 'P_fertilizer': 0.86, 'K_fertilizer': 0.86, 'pesticide': 24.74,
              'seed': 5.57}
    PRICE_WATER = 0.072
    COST_MACHINERY = 151.06
    COST_LABOR = 73.09


class YieldPredictorCPU:
    def __init__(self, model_dir):
        self.model = None
        self.scaler_X = None
        self.feature_order = None
        self._load_all(model_dir)

    def _load_all(self, model_dir):
        try:
            scaler_candidates = [
                os.path.join(model_dir, 'scaler_X.joblib'),
                os.path.join(model_dir, 'scaler_for_xgb.joblib'),
            ]
            model_candidates = [
                os.path.join(model_dir, 'xgb_model.json'),
                os.path.join(model_dir, 'manual_best_xgb_model.json'),
            ]
            scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            if scaler_path is None:
                raise FileNotFoundError(f'No X scaler found in {model_dir}. Tried: {scaler_candidates}')
            if model_path is None:
                raise FileNotFoundError(f'No XGBoost model found in {model_dir}. Tried: {model_candidates}')
            self.scaler_X = joblib.load(scaler_path)
            self.feature_order = self.scaler_X.feature_names_in_
            self.model = xgb.XGBRegressor(tree_method='hist', n_jobs=1)
            self.model.load_model(model_path)
        except Exception as e:
            raise e

    def predict_batch(self, input_features_df: pd.DataFrame) -> np.ndarray:
        input_features_reordered = input_features_df[self.feature_order]
        X_scaled = self.scaler_X.transform(input_features_reordered)
        return self.model.predict(X_scaled)


def calculate_metrics_batch(decision_vars_values, predictor, fixed_vars_df, decision_vars_list, baseline_stats):
    is_population = decision_vars_values.ndim == 2
    if is_population:
        decision_vars_df = pd.DataFrame(decision_vars_values.T, columns=decision_vars_list)
        pop_size = decision_vars_df.shape[0]
        current_batch_df = fixed_vars_df.loc[fixed_vars_df.index.repeat(pop_size)].reset_index(drop=True)
        for var_name in decision_vars_list: current_batch_df[var_name] = decision_vars_df[var_name]
        x_arr = decision_vars_values.T
    else:
        decision_vars_df = pd.DataFrame(decision_vars_values.reshape(1, -1), columns=decision_vars_list)
        current_batch_df = fixed_vars_df.copy()
        for var_name in decision_vars_list: current_batch_df[var_name] = decision_vars_df[var_name]
        x_arr = decision_vars_values.reshape(1, -1)

    current_batch_df['GrowingPeriod'] = current_batch_df['HarvestDate'] - current_batch_df['SowDate']
    yield_pred = predictor.predict_batch(current_batch_df)

    var_map = {name: i for i, name in enumerate(decision_vars_list)}
    density = x_arr[:, var_map['Density']]

    seed_weight_kgha = density * CONFIG.AVG_SEED_WEIGHT_GRAM / 1000000.0

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
         TP * CONFIG.CEC['pesticide'] + seed_weight_kgha * CONFIG.CEC['seed']

    ne_up = ip * CONFIG.NEC['irrigation_power'] + \
            N * CONFIG.NEC['N_fertilizer'] + P * CONFIG.NEC['P_fertilizer'] + \
            K * CONFIG.NEC['K_fertilizer'] + TP * CONFIG.NEC['pesticide'] + \
            seed_weight_kgha * CONFIG.NEC['seed']

    n2o = N * 0.0250 * (44 / 28) * 0.476
    nh3 = N * 0.21 * (17 / 14) * 0.833
    no3 = N * 0.19 * (62 / 14) * 0.238
    ne = ne_up + n2o + nh3 + no3

    CI, NI, WF = ce / (yield_pred + eps), ne / (yield_pred + eps), IWR / (yield_pred + eps)

    st = baseline_stats
    z_ci = (st['CE']['mean'] - CI) / st['CE']['std']
    z_ni = (st['NE']['mean'] - NI) / st['NE']['std']
    z_wf = (st['WF']['mean'] - WF) / st['WF']['std']
    z_tf = (st['TF']['mean'] - TF) / st['TF']['std']
    z_iwr = (st['IWR']['mean'] - IWR) / st['IWR']['std']
    z_tp = (st['TP']['mean'] - TP) / st['TP']['std']
    env_idx = np.mean(np.array([z_ci, z_ni, z_wf, z_tf, z_iwr, z_tp]), axis=0)

    prod_val = yield_pred * CONFIG.PRICES['maize_grain']
    cost = (
                   N * 0.7 + P * 0.86 + K * 0.86 + TP * 24.74 + seed_weight_kgha * 5.57) + IWR * CONFIG.PRICE_WATER + CONFIG.COST_MACHINERY + CONFIG.COST_LABOR
    prof = prod_val - cost
    roi = prof / (cost + eps)

    z_prof = (prof - st['Profit']['mean']) / st['Profit']['std']
    z_roi = (roi - st['ROI']['mean']) / st['ROI']['std']
    econ_idx = np.mean(np.array([z_prof, z_roi]), axis=0)

    return {'Yield': yield_pred, 'Environmental_Index': env_idx, 'Economic_Index': econ_idx}


def objective_function_wrapper(dv, pred, fixed, dl, bs, mode):
    m = calculate_metrics_batch(dv, pred, fixed, dl, bs)
    if mode == 'Yield':
        return -m['Yield']
    elif mode == 'Env':
        return -m['Environmental_Index']
    elif mode == 'Econ':
        return -m['Economic_Index']
    else:
        raise ValueError


def process_chunk(chunk_id, subset_data, config_dict, baseline_stats, mode):
    warnings.filterwarnings("ignore")
    seed_val = config_dict['SEED'];
    np.random.seed(seed_val);
    random.seed(seed_val)
    try:
        predictor = YieldPredictorCPU(config_dict['MODEL_SAVE_DIRECTORY'])
    except Exception as exc:
        raise RuntimeError(f"Failed to load maize model from {config_dict['MODEL_SAVE_DIRECTORY']}") from exc
    results = []

    for idx, (original_idx, row) in enumerate(subset_data.iterrows()):
        try:
            current_fixed = pd.DataFrame([row])
            res = differential_evolution(
                func=objective_function_wrapper, bounds=config_dict['BOUNDS'],
                args=(predictor, current_fixed, config_dict['DECISION_VARIABLES'], baseline_stats, mode),
                strategy=config_dict['DE_STRATEGY'], maxiter=config_dict['DE_MAXITER'],
                popsize=config_dict['DE_POPSIZE'],
                tol=config_dict['TOL_MAPPING'][mode], vectorized=True, polish=False, disp=False, seed=seed_val
            )
            final = calculate_metrics_batch(res.x, predictor, current_fixed, config_dict['DECISION_VARIABLES'],
                                            baseline_stats)
            row_res = {'original_index': original_idx, 'Opt.Yield': final['Yield'][0],
                       'Opt.Environmental_Index': final['Environmental_Index'][0],
                       'Opt.Economic_Index': final['Economic_Index'][0]}
            best_p = dict(zip(config_dict['DECISION_VARIABLES'], res.x))
            for k, v in best_p.items(): row_res[CONFIG.VAR_MAPPING[k]] = v if k not in ['Density', 'SowDate',
                                                                                        'HarvestDate'] else int(
                np.round(v))
            results.append(row_res)
        except Exception as exc:
            print(f"Warning: skipped row {original_idx} for mode {mode}: {exc}")
            continue
    return results


def fill_missing(df):
    for c in CONFIG.DECISION_VARIABLES:
        if c not in df.columns: df[c] = 0.0
    if 'yield' in df.columns and 'Yield' not in df.columns: df.rename(columns={'yield': 'Yield'}, inplace=True)
    return df


def split_dataframe(df, n_chunks):
    index_chunks = np.array_split(np.arange(len(df)), max(int(n_chunks), 1))
    return [df.iloc[idx].copy() for idx in index_chunks if len(idx) > 0]


if __name__ == '__main__':
    np.random.seed(CONFIG.SEED);
    random.seed(CONFIG.SEED)
    print(f"Maize single-objective optimization | Seed={CONFIG.SEED}")

    df_full = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_full = fill_missing(df_full)
    df_std = pd.read_excel(CONFIG.INPUT_EXCEL_PATH)
    df_std = fill_missing(df_std)

    print("  (Model Predicted)...")
    pred = YieldPredictorCPU(CONFIG.MODEL_SAVE_DIRECTORY)
    if 'GrowingPeriod' not in df_std: df_std['GrowingPeriod'] = df_std['HarvestDate'] - df_std['SowDate']
    yld = pred.predict_batch(df_std)

    d = df_std
    den = d['Density']
    sw = den * CONFIG.AVG_SEED_WEIGHT_GRAM / 1000000.0

    N, P, K = d['BasalN'] + d['TopdressingN'], d['BasalP2O5'] + d['TopdressingP2O5'], d['BasalK2O'] + d[
        'TopdressingK2O']
    TF, IWR, TP = N + P + K, d['Irrigation1Amount'] + d['Irrigation2Amount'] + d['Irrigation3Amount'], d['Pesticide']

    ip = IWR * CONFIG.KWH_PER_M3_WATER
    ce = ip * 0.92 + N * 1.53 + P * 1.14 + K * 0.66 + TP * 6.58 + sw * 1.22
    ne_up = ip * 0.12e-3 + N * 0.89e-3 + P * 0.54e-3 + K * 0.03e-3 + TP * 5.02e-3 + sw * 0.88e-3
    n2o = N * 0.0250 * (44 / 28) * 0.476
    nh3 = N * 0.21 * (17 / 14) * 0.833
    no3 = N * 0.19 * (62 / 14) * 0.238
    ne = ne_up + n2o + nh3 + no3

    eps = 1e-6
    CI, NI, WF = ce / (yld + eps), ne / (yld + eps), IWR / (yld + eps)
    rev = yld * CONFIG.PRICES['maize_grain']
    cost = (
                   N * 0.7 + P * 0.86 + K * 0.86 + TP * 24.74 + sw * 5.57) + IWR * CONFIG.PRICE_WATER + CONFIG.COST_MACHINERY + CONFIG.COST_LABOR
    prof, roi = rev - cost, (rev - cost) / (cost + eps)

    stats = {
        'CE': {'mean': CI.mean(), 'std': CI.std()}, 'NE': {'mean': NI.mean(), 'std': NI.std()},
        'WF': {'mean': WF.mean(), 'std': WF.std()}, 'TF': {'mean': TF.mean(), 'std': TF.std()},
        'IWR': {'mean': IWR.mean(), 'std': IWR.std()}, 'TP': {'mean': TP.mean(), 'std': TP.std()},
        'Profit': {'mean': prof.mean(), 'std': prof.std()}, 'ROI': {'mean': roi.mean(), 'std': roi.std()}
    }

    cfg = {k: v for k, v in vars(CONFIG).items() if not k.startswith('__')}
    chunks = split_dataframe(df_full, CONFIG.N_CORES * 4)

    for mode in ['Yield', 'Env', 'Econ']:
        print(f"\n : {mode}")
        start = time.time()
        ts = [(i, chunk, cfg, stats, mode) for i, chunk in enumerate(chunks)]
        with Parallel(n_jobs=CONFIG.N_CORES, return_as='generator') as parallel:
            res_list = []
            for r in tqdm(parallel(delayed(process_chunk)(*t) for t in ts), total=len(ts)): res_list.append(r)
        flat = [x for sub in res_list for x in sub]
        if flat:
            res_df = pd.DataFrame(flat).set_index('original_index')
            out = df_full.copy()
            for c in res_df.columns: out.loc[res_df.index, c] = res_df[c]
            out.to_excel(CONFIG.OUTPUT_PATHS[mode], index=False)
            print(f"Saved: {CONFIG.OUTPUT_PATHS[mode]}")
        print(f"⏱️ : {(time.time() - start) / 60:.2f} min")
