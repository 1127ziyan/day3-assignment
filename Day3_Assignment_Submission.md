# Data Science Deployment — Day 3 Assignment
## Capital Bikeshare: Next-Day Rental Forecasting

**Objective:** Forecast next-day total bike rentals using information available the previous evening, compare two models against a simple benchmark, and track and register the selected model with MLflow.

Each numbered step contains an explanation followed by runnable Python code. Recorded results come from the completed local experiment and have been checked against its MLflow records. Code cells have no fabricated execution outputs. Running the notebook creates new runs and model versions.

**Submission links**
- GitHub repository: [1127ziyan/day3-assignment](https://github.com/1127ziyan/day3-assignment).
- Remote MLflow model page: [bike_demand_forecast — Version 1](https://day3-assignment-mlflow.onrender.com/#/models/bike_demand_forecast/versions/1).
- Original local registered model URI: `models:/bike_demand_forecast/1`.

A model URI identifies an object inside MLflow; it is not a browser URL. The HTTPS model-version page above is publicly accessible for instructor review. The hosted copy preserves the original fitted Ridge pipeline and has its own run identity.

**Sources**
- Course assignment: DSD_Day3_Learner.pdf, PDF pages 19–21.
- [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
- [MLflow Model Registry documentation](https://mlflow.org/docs/latest/ml/model-registry/).

## 1. Set up Python

Open this notebook in VS Code, enable the Python and Jupyter extensions, and select a Python kernel. Install the dependencies below only if they are missing. Restart the kernel after installation if requested, then run the notebook from the beginning.

The recorded experiment used Python 3.14.6. The pinned core package versions below match that environment.

```python
# Install missing dependencies, then restart the kernel if requested.
# %pip install "mlflow==3.17.0" "scikit-learn==1.9.0" "pandas==3.0.3" "numpy==2.4.6" "skops==0.16.0"
```

```python
import io
import os
import json
import sqlite3
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import sklearn
import mlflow
import mlflow.sklearn

from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.tracking import MlflowClient

print("Python libraries ready.")
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("MLflow:", mlflow.__version__)
```

## 2. Start MLflow

MLflow stores experiment parameters, metrics, run metadata, data snapshots, and fitted models. Configure a local SQLite database and artifact directory inside this project. Creating the directory before opening the database avoids the earlier `unable to open database file` error.

To upload new experiments to a hosted server, set `MLFLOW_TRACKING_URI` to its HTTPS tracking endpoint before starting the notebook kernel. Supply credentials through environment variables, never notebook source. For remote experiments, the server selects the artifact location; a local file path would be inaccessible to the teacher.

The optional local UI below is useful for inspection. A `127.0.0.1` address is accessible only on the computer running it and cannot serve as the submission's remote model link.

The submitted Render service is a public, read-only snapshot for instructor inspection and model retrieval. Keep the default local mode when rerunning this notebook. Use a separately authenticated writable server if you want to log new experiments remotely.

```python
PROJECT_DIR = Path.cwd()
store_dir = PROJECT_DIR / "local_mlflow"
artifact_dir = store_dir / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)

db_path = store_dir / "day3_mlflow.db"
remote_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
if remote_tracking_uri:
    if not remote_tracking_uri.startswith("https://"):
        raise ValueError("Use an HTTPS endpoint for remote MLflow tracking.")
    tracking_uri = remote_tracking_uri
else:
    with sqlite3.connect(str(db_path)):
        pass
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", tracking_uri))
experiment_name = "Day3_Bike_Demand"
if mlflow.get_experiment_by_name(experiment_name) is None:
    experiment_options = {"name": experiment_name}
    if not remote_tracking_uri:
        experiment_options["artifact_location"] = artifact_dir.as_uri()
    mlflow.create_experiment(**experiment_options)
mlflow.set_experiment(experiment_name)
print("Tracking mode:", "remote" if remote_tracking_uri else "local")
```

```python
# Optional: change False to True to start the local MLflow UI.
START_MLFLOW_UI = False

if START_MLFLOW_UI and not remote_tracking_uri:
    import subprocess
    import sys

    with (store_dir / "mlflow_ui.log").open("a") as log_file:
        ui_process = subprocess.Popen(
            [
                sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", tracking_uri,
                "--host", "127.0.0.1",
                "--port", "5001",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    print("MLflow UI: http://127.0.0.1:5001")
    print("Server log:", store_dir / "mlflow_ui.log")
else:
    print("UI not started. Experiment tracking is ready.")
```

## 3. Add the dataset

Use the daily records in `day.csv` from the UCI Bike Sharing dataset, rather than the hourly records in `hour.csv`. Download the official archive once and cache the daily CSV inside the project. Subsequent executions reuse the cache.

Each row represents one day. The target column, `cnt`, contains the total daily bike rental count.

```python
data_dir = store_dir / "data"
data_dir.mkdir(exist_ok=True)
raw_csv = data_dir / "day.csv"

dataset_url = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike%2Bsharing%2Bdataset.zip"
)

if not raw_csv.exists():
    with urlopen(dataset_url, timeout=60) as response:
        zip_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        daily_file = next(
            name for name in archive.namelist()
            if name.split("/")[-1] == "day.csv"
        )
        with archive.open(daily_file) as file:
            df = pd.read_csv(file)
    df.to_csv(raw_csv, index=False)
else:
    df = pd.read_csv(raw_csv)

df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values("dteday").reset_index(drop=True)
display(df.head())
```

## 4. Inspect the data and define the task

Forecast the target day's `cnt` using calendar information and historical rental counts. Month, weekday, holiday status, and working-day status are known ahead of time.

Target-day observed weather is unavailable at the previous evening's forecast time. The target-day `casual` and `registered` columns are also excluded because `cnt = casual + registered`; using them would reveal the target.

**Recorded inspection:** 731 rows and 16 columns; dates from 2011-01-01 through 2012-12-31; no duplicate dates and no missing values. Check that dates are sorted and consecutive before interpreting row shifts as day lags.

```python
print("Shape:", df.shape)
print("Date range:", df["dteday"].min(), "to", df["dteday"].max())
print("Duplicate dates:", df["dteday"].duplicated().any())
print("Missing values:", df.isna().sum().sum())
print("cnt = casual + registered:",
      df["cnt"].eq(df["casual"] + df["registered"]).all())

assert not df["dteday"].duplicated().any()
assert df["dteday"].diff().dropna().eq(pd.Timedelta(days=1)).all()
```

## 5. Create historical features and the benchmark

This experiment interprets the assignment's wording, “use yesterday's rentals to forecast tomorrow,” relative to a decision day. If the target day is $t$, the decision day is $t-1$, so the benchmark uses $t-2$:

$$
\hat y_t^{baseline} = y_{t-2}.
$$

This is the baseline convention used throughout the recorded experiment. A one-day persistence forecast would be a different benchmark and should not be mixed with these results.

Create two lag features and a seven-day historical average:

- `rentals_lag2`: the rental count two days before the target date.
- `rentals_lag7`: the rental count one week before the target date.
- `rentals_mean7`: the average count from target day minus eight through target day minus two.

$$
mean7_t = \frac{1}{7}\sum_{k=2}^{8} y_{t-k}.
$$

Apply `shift(2)` before `rolling(7)` so the average ends at the latest day permitted by this experiment's convention. The first eight rows have insufficient history and will be excluded from modelling.

```python
data = df.copy()

data["baseline_pred"] = data["cnt"].shift(2)
data["rentals_lag2"] = data["cnt"].shift(2)
data["rentals_lag7"] = data["cnt"].shift(7)
data["rentals_mean7"] = (
    data["cnt"].shift(2).rolling(window=7, min_periods=7).mean()
)

display(data[
    [
        "dteday", "cnt", "baseline_pred",
        "rentals_lag2", "rentals_lag7", "rentals_mean7",
    ]
].head(12))
```

## 6. Split the dataset chronologically

Train on earlier dates, select the model on a later validation period, and evaluate the selected model on the held-out test period. This ordering reflects the use of past observations to forecast future demand.

Forecasts are generated daily as history becomes available. This evaluation does not represent forecasting an entire quarter at its first date.

**Recorded split**

| Split | Rows | Start | End |
|---|---:|---|---|
| Train | 539 | 2011-01-09 | 2012-06-30 |
| Validation | 92 | 2012-07-01 | 2012-09-30 |
| Test | 92 | 2012-10-01 | 2012-12-31 |

The date boundaries are experimental design choices. Removing the first eight rows leaves 723 modelling observations.

```python
feature_cols = [
    "mnth", "weekday", "holiday", "workingday",
    "rentals_lag2", "rentals_lag7", "rentals_mean7",
]

model_data = data.dropna(
    subset=feature_cols + ["cnt", "baseline_pred"]
).copy()

train = model_data[model_data["dteday"] < "2012-07-01"].copy()
valid = model_data[
    model_data["dteday"].between("2012-07-01", "2012-09-30")
].copy()
test = model_data[model_data["dteday"] >= "2012-10-01"].copy()

split_summary = pd.DataFrame([
    {
        "split": name,
        "rows": len(part),
        "start": part["dteday"].min(),
        "end": part["dteday"].max(),
    }
    for name, part in [("train", train), ("validation", valid), ("test", test)]
])

X_train, y_train = train[feature_cols].copy(), train["cnt"].copy()
X_valid, y_valid = valid[feature_cols].copy(), valid["cnt"].copy()

display(split_summary)
```

## 7. Evaluate the benchmark

Use validation **MAE as the primary model-selection metric** and report RMSE as a supplementary metric. Both measure error in daily rental counts; smaller values are better.

$$
MAE = \frac{1}{n}\sum_i |y_i-\hat y_i|,
\qquad
RMSE = \sqrt{\frac{1}{n}\sum_i (y_i-\hat y_i)^2}.
$$

MAE answers: “How many rentals is the forecast wrong by on an average day?” It provides a direct interpretation without assuming a disproportionate cost for extreme errors. RMSE gives greater weight to large errors and provides a complementary view of reliability. If a future operating policy assigns especially high costs to large shortages, the selection metric should be reconsidered before model selection.

**Recorded validation benchmark:** MAE = 900.89; RMSE = 1163.55.

```python
def regression_metrics(y_true, predictions):
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
    }

baseline_valid = regression_metrics(y_valid, valid["baseline_pred"])
display(pd.DataFrame([{
    "model": "Baseline (lag2)",
    "MAE": baseline_valid["mae"],
    "RMSE": baseline_valid["rmse"],
}]).round(2))
```

## 8. Train two models and track the experiments

**Ridge regression** learns a weighted combination of inputs and uses L2 regularization to constrain the coefficients. The regularization strength is `alpha = 1.0`.

**Random forest regression** averages multiple decision trees and can represent nonlinear relationships. Use 300 trees, maximum depth 8, at least 5 observations per leaf, and random seed 42.

Encode calendar categories with one-hot encoding and standardize historical numeric features. Fit preprocessing only on the training set. Both models use identical observations and input columns. Random forests generally do not need standardization, but this transformation preserves numeric ordering and allows both pipelines to share preprocessing.

The tracking function records model parameters, feature names, date ranges, package versions, data snapshots, training and validation metrics, and the fitted preprocessing-plus-model pipeline.

Use `skops` serialization. For this locally trained random forest, explicitly trust `sklearn.tree._tree.Tree`, its internal tree structure. This resolves the serialization error encountered in the completed experiment without trusting unrelated types.

```python
calendar_cols = ["mnth", "weekday", "holiday", "workingday"]
history_cols = ["rentals_lag2", "rentals_lag7", "rentals_mean7"]

preprocessor = ColumnTransformer([
    (
        "calendar",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        calendar_cols,
    ),
    ("history", StandardScaler(), history_cols),
])

ridge_params = {"alpha": 1.0}
rf_params = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

ridge_model = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("regressor", Ridge(**ridge_params)),
])
rf_model = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("regressor", RandomForestRegressor(**rf_params)),
])
```

```python
def train_and_track(model, model_name, settings, trusted_types=None):
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "model_type": model_name,
            **settings,
            "features": ",".join(feature_cols),
            "train_start": str(train["dteday"].min().date()),
            "train_end": str(train["dteday"].max().date()),
            "valid_start": str(valid["dteday"].min().date()),
            "valid_end": str(valid["dteday"].max().date()),
        })
        mlflow.set_tags({
            "task": "daily_rental_forecasting",
            "baseline_rule": "target_day_minus_2",
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "mlflow_version": mlflow.__version__,
            "dataset_source": "UCI Bike Sharing, day.csv, 2011-2012",
        })
        mlflow.log_dict(
            {
                "features": feature_cols,
                "splits": split_summary.astype(str).to_dict("records"),
                "preprocessing": {
                    "calendar": "OneHotEncoder",
                    "history": "StandardScaler",
                },
            },
            "run_context.json",
        )

        snapshot_cols = ["dteday"] + feature_cols + ["cnt"]
        mlflow.log_text(
            train[snapshot_cols].to_csv(index=False), "data/train.csv"
        )
        mlflow.log_text(
            valid[snapshot_cols + ["baseline_pred"]].to_csv(index=False),
            "data/validation.csv",
        )
        mlflow.log_artifact(str(raw_csv), artifact_path="data/raw")

        model.fit(X_train, y_train)
        val_scores = regression_metrics(y_valid, model.predict(X_valid))
        train_scores = regression_metrics(y_train, model.predict(X_train))

        scores = {
            "train_mae": train_scores["mae"],
            "train_rmse": train_scores["rmse"],
            "val_mae": val_scores["mae"],
            "val_rmse": val_scores["rmse"],
            "baseline_val_mae": baseline_valid["mae"],
            "baseline_val_rmse": baseline_valid["rmse"],
        }
        mlflow.log_metrics(scores)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(3),
            serialization_format="skops",
            skops_trusted_types=trusted_types,
        )
        run_id = run.info.run_id

    return {
        "name": model_name,
        "model": model,
        "model_info": model_info,
        "run_id": run_id,
        "metrics": scores,
    }

ridge_result = train_and_track(ridge_model, "Ridge", ridge_params)
rf_result = train_and_track(
    rf_model, "Random Forest", rf_params,
    trusted_types=["sklearn.tree._tree.Tree"],
)

print("Ridge Run ID:", ridge_result["run_id"])
print("Random Forest Run ID:", rf_result["run_id"])
print("Random Forest training MAE:",
      round(rf_result["metrics"]["train_mae"], 2))
```

## 9. Compare models and select the candidate

Select the candidate with the lowest validation MAE. Keep the test period out of model selection.

**Recorded validation results**

| Model | MAE | RMSE | MAE improvement vs. baseline |
|---|---:|---:|---:|
| Baseline (lag2) | 900.89 | 1163.55 | 0.00% |
| Ridge | 834.22 | 1014.61 | 7.40% |
| Random Forest | 1243.75 | 1410.43 | -38.06% |

Ridge performs best on both reported validation metrics. The random forest has training MAE 501.65 but worse validation performance. This gap may reflect overfitting, differences between time periods, or both; these scores alone do not establish the cause.

Percentage improvement is `(baseline MAE - model MAE) / baseline MAE × 100`. A negative value means the model has higher error than the benchmark. It is not a classification accuracy score.

```python
candidate_results = [ridge_result, rf_result]

comparison = pd.DataFrame([
    {
        "model": "Baseline (lag2)",
        "MAE": baseline_valid["mae"],
        "RMSE": baseline_valid["rmse"],
    },
    *[
        {
            "model": result["name"],
            "MAE": result["metrics"]["val_mae"],
            "RMSE": result["metrics"]["val_rmse"],
        }
        for result in candidate_results
    ],
])
comparison["MAE_improvement_%"] = (
    (baseline_valid["mae"] - comparison["MAE"])
    / baseline_valid["mae"] * 100
)

selected = min(
    candidate_results, key=lambda result: result["metrics"]["val_mae"]
)
print("Selected candidate:", selected["name"])
display(comparison.sort_values("MAE").round(2))
```

## 10. Register and retrieve the selected model

Register the same fitted candidate evaluated on validation data. Registration organizes the model under a name and version and links it to its source experiment. It does not retrain the model or create a prediction-serving endpoint.

**Original local registration**

- Name: `bike_demand_forecast`
- Version: `1`
- Model URI: `models:/bike_demand_forecast/1`
- Source run: `d6fc343b23934fba8bfb053c5413f4e8`

A fresh execution uses the new run ID and returned version instead of forcing the original local identity. Remote registration must be verified separately; the original local version is not evidence of a completed cloud upload.

```python
registered = mlflow.register_model(
    model_uri=selected["model_info"].model_uri,
    name="bike_demand_forecast",
)
registered_uri = f"models:/{registered.name}/{registered.version}"

assert registered.run_id == selected["run_id"], "Source run mismatch"

registered_model = mlflow.sklearn.load_model(registered_uri)

print("Name:", registered.name)
print("Version:", registered.version)
print("Source Run:", registered.run_id)
print("URI:", registered_uri)
```

## 11. Evaluate on the test set and save the assessment

Compare the registered candidate and baseline on the same test dates. Check that reloading the saved model produces predictions consistent with the fitted pipeline. Record the final test metrics, predictions, assessment, and model-version check.

**Recorded test results**

| Model | MAE | RMSE | MAE improvement |
|---|---:|---:|---:|
| Baseline (lag2) | 1297.01 | 1821.98 | 0.00% |
| Registered Ridge | 1168.61 | 1537.39 | 9.90% |

The recorded model passed the reload-consistency check and achieved lower test MAE than the baseline. This is an offline assessment, not a guarantee of production performance.

```python
X_test, y_test = test[feature_cols].copy(), test["cnt"].copy()
selected_test_pred = registered_model.predict(X_test)

np.testing.assert_allclose(
    selected_test_pred,
    selected["model"].predict(X_test),
)

baseline_test = regression_metrics(y_test, test["baseline_pred"])
selected_test = regression_metrics(y_test, selected_test_pred)

test_results = pd.DataFrame([
    {
        "model": "Baseline (lag2)",
        "MAE": baseline_test["mae"],
        "RMSE": baseline_test["rmse"],
    },
    {
        "model": f"Registered {selected['name']}",
        "MAE": selected_test["mae"],
        "RMSE": selected_test["rmse"],
    },
])
test_results["MAE_improvement_%"] = (
    (baseline_test["mae"] - test_results["MAE"])
    / baseline_test["mae"] * 100
)

passed_benchmark = selected_test["mae"] < baseline_test["mae"]

with mlflow.start_run(run_id=selected["run_id"]):
    mlflow.log_metrics({
        "test_mae": selected_test["mae"],
        "test_rmse": selected_test["rmse"],
        "baseline_test_mae": baseline_test["mae"],
        "baseline_test_rmse": baseline_test["rmse"],
    })
    test_report = test[
        ["dteday"] + feature_cols + ["cnt", "baseline_pred"]
    ].copy()
    test_report["model_pred"] = selected_test_pred
    mlflow.log_text(
        test_report.to_csv(index=False),
        "evaluation/test_predictions.csv",
    )
    mlflow.log_dict(
        {
            "selection_metric": "validation_mae",
            "validation_comparison": comparison.to_dict("records"),
            "registered_model_uri": registered_uri,
            "test_mae_beats_baseline": bool(passed_benchmark),
        },
        "evaluation/final_assessment.json",
    )

client = MlflowClient()
client.set_model_version_tag(
    name=registered.name,
    version=registered.version,
    key="test_benchmark_check",
    value="passed" if passed_benchmark else "failed",
)

display(test_results.round(2))
print("Predictions are consistent after reload.")
print("Test MAE benchmark check:",
      "passed" if passed_benchmark else "failed")
```

## 12. Summarize the assignment

This experiment forecasts next-day total bike rentals using calendar information and historical rental features. Data were split chronologically. The benchmark uses the rental count from two days before the target date, following the interpretation stated in Step 5. Target-day observed weather and rental-count components were excluded.

Ridge achieved the lowest validation MAE: 834.22, compared with 900.89 for the benchmark and 1243.75 for the random forest. It was therefore selected. On the held-out test period, the registered Ridge model achieved MAE 1168.61 and RMSE 1537.39, compared with benchmark values of 1297.01 and 1821.98. Test MAE decreased by approximately 9.90%.

MLflow preserved experiment settings, data snapshots, metrics, and model artifacts. The original selected model was registered locally as `bike_demand_forecast`, version 1, linked to source run `d6fc343b23934fba8bfb053c5413f4e8`. Reloading it produced predictions consistent with the original model.

These offline results support further operational evaluation. They do not establish production performance or realized business savings. Higher test errors for both forecasts suggest the later period may be more difficult, but do not identify a specific drift mechanism.

**Reproducibility:** Save this notebook in VS Code. The optional cell below attaches the saved document to the selected run. Keep the notebook in the project root or update `NOTEBOOK_PATH` after moving it.

```python
NOTEBOOK_PATH = PROJECT_DIR / "Day3_Assignment_Submission.ipynb"

if NOTEBOOK_PATH.is_file():
    with mlflow.start_run(run_id=selected["run_id"]):
        mlflow.log_artifact(str(NOTEBOOK_PATH), artifact_path="submission")
    print("Notebook attached to Run:", selected["run_id"])
else:
    print("Update NOTEBOOK_PATH to the saved notebook location.")
```

## 13. Access the hosted MLflow model

The original selected Ridge pipeline is registered as **bike_demand_forecast, Version 1** on the hosted MLflow server.

- [Open the registered model version](https://day3-assignment-mlflow.onrender.com/#/models/bike_demand_forecast/versions/1).
- Tracking endpoint: `https://day3-assignment-mlflow.onrender.com`.
- Model loading URI: `models:/bike_demand_forecast/1`.
- Original local source run: `d6fc343b23934fba8bfb053c5413f4e8`.

The browser URL opens a specific model version. The tracking endpoint connects Python to the server. The model URI selects the model within that server. The hosted run has a new ID and a provenance tag linking it to the original local run; the fitted model is copied without retraining.

The public model-version page was checked on 8 October 2026. The hosted service permits reading and model downloads; rerunning the training cells uses the local store by default. Render's Free service may take approximately one minute to wake after inactivity. A manual rebuild recreates hosted run IDs, so recheck provenance after rebuilding.

The following cell reads the submitted link from `submission_links.json`.

**External verification:** An independent HTTPS client retrieved registered Version 1 without a login and compared its predictions on all 92 test rows with the original saved predictions. They matched within numerical tolerance (`rtol=1e-10`). Verified test MAE: **1168.61**; test RMSE: **1537.39**. Details and the hosted run ID are recorded in `remote_model_verification.json`.

```python
from IPython.display import Markdown

# The default link also works when opening this notebook as a standalone file.
default_model_page_url = "https://day3-assignment-mlflow.onrender.com/#/models/bike_demand_forecast/versions/1"
links_path = PROJECT_DIR / "submission_links.json"
submission_links = json.loads(links_path.read_text()) if links_path.exists() else {}
model_page_url = submission_links.get("mlflow_model_page_url") or default_model_page_url
display(Markdown(f"[Registered MLflow model]({model_page_url})"))
```
