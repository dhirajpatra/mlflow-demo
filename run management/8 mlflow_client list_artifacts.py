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

print(f"Run tags: {run.data.tags}")
print(f"Experiment id: {run.info.experiment_id}")
print(f"Run id: {run.info.run_id}")
print(f"Run name: {run.info.run_name}")
print(f"lifecycle_stage: {run.info.lifecycle_stage}")
print(f"status: {run.info.status}")


artifacts = client.list_artifacts(run.info.run_id)

for artifact in artifacts:
    print(f"artifact: {artifact.path}")
    print(f"size: {artifact.file_size}")