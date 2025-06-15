import mlflow
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess()
model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
print(classification_report(y_test, model.predict(X_test)))
