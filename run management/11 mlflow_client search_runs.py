import mlflow
import joblib
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

runs = client.search_runs(
    experiment_ids=["6", "10", "12", "14", "22"],
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["run_id ASC"],
    filter_string="run_name = 'Mlflow Client Run'"
)

for run in runs:
    print(f"Run name: {run.info.run_name}, Run ID: {run.info.run_id}")