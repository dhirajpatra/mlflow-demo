import mlflow
import joblib
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

run = client.get_run("622a509ad9594658960ccf0088274daa")

metrics = client.get_metric_history(run.info.run_id, 'rmse')

for metric in metrics:
    print(f"Step: {metric.step}, Timestamp: {metric.timestamp}, Value: {metric.value}")