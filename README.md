# Day 3 Assignment: Bike Demand Forecasting

This submission compares Ridge regression and a random forest with a lag-2 benchmark, tracks experiments in MLflow, and registers the selected model.

## Open in VS Code

1. Open this repository folder.
2. Install dependencies from `requirements.txt` in a Python environment. The recorded experiment used Python 3.14.6.
3. Open `Day3_Assignment_Submission.ipynb`, select that environment as the kernel, and run cells in order.
4. Read the explanations and recorded results alongside each numbered step.

A Markdown copy is included for reading without a notebook viewer. Fresh executions create new MLflow runs and versions; the original results are labelled as recorded results.

## Recorded results

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Baseline (lag2) | 900.89 | 1163.55 |
| Ridge | 834.22 | 1014.61 |
| Random Forest | 1243.75 | 1410.43 |

Ridge was selected using validation MAE. Its test MAE was 1168.61, compared with 1297.01 for the baseline, a 9.90% reduction. Test RMSE was 1537.39, compared with 1821.98 for the baseline.

## MLflow and submission links

`submission_links.json` records the verified repository and remote model webpage URLs. The public repository is [1127ziyan/day3-assignment](https://github.com/1127ziyan/day3-assignment). The published model is [bike_demand_forecast, Version 1](https://day3-assignment-mlflow.onrender.com/#/models/bike_demand_forecast/versions/1). The HTTPS page is publicly accessible to the instructor.

The submitted Render service is a public, read-only model snapshot. Keep the default local mode when rerunning this notebook. To log new experiments on a separate authenticated writable server, set `MLFLOW_TRACKING_URI` to that server's HTTPS endpoint before starting the kernel. Provide credentials only through the provider's authentication mechanism or environment variables. The notebook leaves artifact-store configuration to the remote server.

Model registration and upload enable inspection and retrieval. They do not themselves create a live prediction API.

## Data and model provenance

Daily data are from the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the notebook downloads the official archive when needed.

`model_export.zip` contains the original selected Ridge model and recorded evaluation snapshots. Extract the archive into a `model_export` folder to load it locally. The same fitted pipeline is registered on the public MLflow server without retraining. The original source run ID is `d6fc343b23934fba8bfb053c5413f4e8`. The original local registered name and version are `bike_demand_forecast`, version 1.

The public model was loaded through an independent HTTPS client without a login. Its predictions matched the original saved predictions on all 92 test rows (`rtol=1e-10`). See `remote_model_verification.json` for verification results and hosted provenance. The service uses Render's Free plan; initial access after inactivity may take approximately one minute.
