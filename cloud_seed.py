"""Build a hosted MLflow snapshot from the original selected model export."""
import json
import os
import shutil
import sqlite3
import zipfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd


def seed_store(project_dir, storage_dir):
    project_dir, storage_dir = Path(project_dir), Path(storage_dir)
    export = storage_dir / 'source_model'
    export.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(project_dir / 'model_export.zip') as archive:
        archive.extractall(export)
    artifacts = storage_dir / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    uri = f"sqlite:///{(storage_dir / 'mlflow.db').resolve().as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment_id = mlflow.create_experiment(
        'Day3_Bike_Demand', artifact_location=artifacts.resolve().as_uri()
    )
    model = mlflow.sklearn.load_model(str(export))
    train = pd.read_csv(export / 'data/train.csv')
    valid = pd.read_csv(export / 'data/validation.csv')
    test = pd.read_csv(export / 'evaluation/test_predictions.csv')
    features = ['mnth', 'weekday', 'holiday', 'workingday',
                'rentals_lag2', 'rentals_lag7', 'rentals_mean7']
    predictions = model.predict(test[features])
    np.testing.assert_allclose(predictions, test['ridge_pred'], rtol=1e-10)
    test_mae = float(np.abs(test['cnt'].to_numpy() - predictions).mean())
    test_rmse = float(np.sqrt(((test['cnt'].to_numpy() - predictions)**2).mean()))
    valid_predictions = model.predict(valid[features])
    val_mae = float(np.abs(valid['cnt'].to_numpy() - valid_predictions).mean())
    val_rmse = float(np.sqrt(((valid['cnt'].to_numpy() - valid_predictions)**2).mean()))
    with mlflow.start_run(experiment_id=experiment_id, run_name='Original Ridge cloud copy') as run:
        mlflow.log_params({'model_type': 'Ridge', 'alpha': 1.0,
                           'features': ','.join(features),
                           'train_start': '2011-01-09', 'train_end': '2012-06-30'})
        mlflow.set_tags({'original_local_run_id': 'd6fc343b23934fba8bfb053c5413f4e8',
                         'model_origin': 'original fitted pipeline; not retrained',
                         'selection_metric': 'validation_mae'})
        mlflow.log_metrics({'val_mae':val_mae,'val_rmse':val_rmse,
                            'test_mae':test_mae,'test_rmse':test_rmse,
                            'baseline_val_mae':900.8913043478261,
                            'baseline_val_rmse':1163.547891430419,
                            'baseline_test_mae':1297.0108695652175,
                            'baseline_test_rmse':1821.984086338178})
        for subfolder in ['data', 'evaluation']:
            mlflow.log_artifacts(str(export / subfolder), artifact_path=subfolder)
        mlflow.log_artifact(str(project_dir / 'Day3_Assignment_Submission.ipynb'),
                            artifact_path='submission')
        info = mlflow.sklearn.log_model(model, name='model',
            input_example=train[features].head(3), serialization_format='skops')
        run_id = run.info.run_id
    registered = mlflow.register_model(info.model_uri, 'bike_demand_forecast')
    loaded = mlflow.sklearn.load_model(f'models:/{registered.name}/{registered.version}')
    np.testing.assert_allclose(loaded.predict(test[features]), predictions)
    mlflow.MlflowClient().set_model_version_tag(registered.name, registered.version,
                                               'test_benchmark_check', 'passed')
    result = {'name':registered.name,'version':registered.version,'run_id':run_id,
              'source_local_run_id':'d6fc343b23934fba8bfb053c5413f4e8',
              'test_mae':test_mae,'test_rmse':test_rmse}
    (storage_dir / 'cloud_model_identity.json').write_text(json.dumps(result,indent=2))
    # Convert this completed snapshot to server-proxied artifact addresses.
    # All runs are finished, and the build process exits before the public server starts.
    file_root = artifacts.resolve().as_uri()
    with sqlite3.connect(storage_dir / 'mlflow.db') as connection:
        for table, field in [('experiments', 'artifact_location'),
                             ('runs', 'artifact_uri'),
                             ('logged_models', 'artifact_location'),
                             ('model_versions', 'source'),
                             ('model_versions', 'storage_location')]:
            connection.execute(
                f'UPDATE {table} SET {field} = REPLACE({field}, ?, ?) '
                f'WHERE {field} LIKE ?',
                (file_root, 'mlflow-artifacts:', file_root + '%'),
            )
    for descriptor in artifacts.rglob('MLmodel'):
        descriptor.write_text(descriptor.read_text().replace(file_root, 'mlflow-artifacts:'))
    shutil.rmtree(export)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    seed_store(Path('/app'), Path('/app/seed'))
