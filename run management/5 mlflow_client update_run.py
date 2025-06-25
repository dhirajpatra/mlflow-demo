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

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Read the wine-quality csv file from the URL
data = pd.read_csv("red-wine-quality.csv")

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = 0.4
l1_ratio = 0.4

lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

predicted_qualities = lr.predict(test_x)

(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

joblib.dump(lr, "linear_regression.pkl")

alpha = client.log_param(run.info.run_id, "alpha", alpha)
l1_ratio = client.log_param(run.info.run_id, "l1_ratio", l1_ratio)

client.log_metric(run.info.run_id, "rmse", rmse)
client.log_metric(run.info.run_id, "mae", mae)
client.log_metric(run.info.run_id, "r2", r2)

client.log_artifact(run.info.run_id, "linear_regression.pkl")
client.log_artifact(run.info.run_id,"red-wine-quality.csv")

print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

client.update_run(run.info.run_id, status='FINISHED', name='Mlflow Client Run')
