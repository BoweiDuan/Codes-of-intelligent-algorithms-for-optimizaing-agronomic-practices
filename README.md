# Codes of Intelligent Algorithms for Optimizing Agronomic Practices

Code and trained model artifacts for farm-level sustainability diagnosis and multi-objective optimization in wheat-maize systems in the North China Plain.

## Repository Contents

- `data_demo/`: small demo Excel datasets with the same schema as the farm-household survey data.
- `models/saved_maize_models_xgb/`: trained XGBoost maize surrogate model and scaler.
- `models/saved_wheat_models_ftt/`: trained FTTransformer wheat surrogate model and scalers.
- `maize_xgboost_train.py`: maize surrogate model training pipeline.
- `wheat_fttransformer_train.py`: wheat surrogate model training pipeline.
- `optimization codes/`: single-objective differential evolution and three-objective NSGA-III optimization scripts.
- `farmer typology/`: baseline typology clustering and post-optimization typology transition analysis.

## Data Availability

The original farm-household survey dataset is not publicly released due to confidentiality and data-use restrictions. This repository includes demo datasets and trained model artifacts so that the workflow can be inspected and run with data that follow the same schema.

Required yield column: `Yield`.

Main management and covariate columns:

- Wheat: `Seed`, `SowDate`, `HarvestDate`, `GrowingPeriod`, fertilizer, irrigation, pesticide, soil, and climate variables.
- Maize: `Density`, `SowDate`, `HarvestDate`, `GrowingPeriod`, fertilizer, irrigation, pesticide, soil, and climate variables.

See `data_demo/wheat.xlsx` and `data_demo/maize.xlsx` for the exact column names.

## Software Environment

Recommended Python version used for the manuscript code: **Python 3.12**.

Core dependencies:

```bash
pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 scipy==1.14.1 joblib==1.4.2 openpyxl==3.1.5 tqdm==4.66.5 matplotlib==3.9.2 xgboost==2.1.3 pymoo==0.6.1.3
```

Additional wheat/FTTransformer dependencies:

```bash
pip install torch==2.2.0 rtdl==0.14.0.dev5
```

Important note: the wheat model uses `rtdl`. The public PyPI combination above supports Python 3.12. If the exact manuscript environment used a later local development build such as `rtdl` dev7, install that source build instead of `rtdl==0.14.0.dev5`. Maize scripts do not require `torch` or `rtdl`.

## Quick Maize Demo

The maize scripts use the model artifacts under `models/saved_maize_models_xgb/` by default.

To run a lightweight single-objective demo:

```bash
MAIZE_N_CORES=1 MAIZE_DE_MAXITER=1 MAIZE_DE_POPSIZE=5 python "optimization codes/maize_single_objective_optimize.py"
```

To run a lightweight multi-objective demo:

```bash
MAIZE_CPU_CORES=1 MAIZE_NSGA_N_GEN=1 MAIZE_NSGA_POP_SIZE=28 python "optimization codes/maize_multi_objective_optimize.py"
```

Full optimization settings are the script defaults:

- Single-objective DE: `DE_MAXITER=500`, `DE_POPSIZE=200`.
- Multi-objective NSGA-III: `N_GEN=150`, `POP_SIZE=92`.

Outputs are written to `outputs/` and `results/`.

## Wheat Scripts

The wheat scripts use `models/saved_wheat_models_ftt/` by default and require the FTTransformer dependencies above.

The wheat optimization scripts are GPU-oriented because the original workflow parallelized FTTransformer inference across CUDA devices. They fail fast with a clear error if no CUDA device is available.

Lightweight settings can be supplied in the same way:

```bash
WHEAT_DE_MAXITER=1 WHEAT_DE_POPSIZE=5 python "optimization codes/wheat_single_objective_optimize.py"
WHEAT_NSGA_N_GEN=1 WHEAT_NSGA_POP_SIZE=28 python "optimization codes/wheat_multi_objective_optimize.py"
```

## Training

To retrain the maize model using the demo data:

```bash
python maize_xgboost_train.py
```

To retrain the wheat model:

```bash
python wheat_fttransformer_train.py
```

Training outputs are saved under `models/saved_maize_models_xgb/seed_<seed>/` and `models/saved_wheat_models_ftt/seed_<seed>/`.

## Environment Variables

Common overrides:

- `DATA_DIR`, `MAIZE_FILE`, `WHEAT_FILE`: training data location.
- `MAIZE_MODEL_DIR`, `WHEAT_MODEL_DIR`: model artifact directories.
- `MAIZE_OPT_INPUT`, `WHEAT_OPT_INPUT`: optimization input files.
- `MAIZE_DE_MAXITER`, `MAIZE_DE_POPSIZE`, `MAIZE_N_CORES`: maize single-objective runtime controls.
- `MAIZE_NSGA_N_GEN`, `MAIZE_NSGA_POP_SIZE`, `MAIZE_CPU_CORES`: maize multi-objective runtime controls.
- `WHEAT_DE_MAXITER`, `WHEAT_DE_POPSIZE`, `WHEAT_NSGA_N_GEN`, `WHEAT_NSGA_POP_SIZE`: wheat optimization runtime controls.

## Reproducibility Notes

All scripts set fixed random seeds where applicable. The demo data are intended to verify code execution and data schema, not to reproduce manuscript-scale statistics from the confidential survey dataset.

Maize seed weight is computed consistently as:

```text
seed_weight_kg_ha = Density * 344.0 / 1,000,000
```
