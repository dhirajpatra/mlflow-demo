import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import logging
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--penality", type=str, required=False, default="l2")
parser.add_argument("--C", type=float, required=False, default=1.0)
args = parser.parse_args()

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the dataset
    data = pd.read_csv('iris.csv')
    os.mkdir("data/")
    data.to_csv("data/iris.csv", index=False)
    # Split the data into training and test sets.
    train, test = train_test_split(data)

    # Data Preprocessing
    le = LabelEncoder()
    train['variety'] = le.fit_transform(train['variety'])
    test['variety'] = le.fit_transform(test['variety'])

    # Storing the training and testing dataset
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")

    # Split
    train_x = train.drop(["variety"], axis=1)
    test_x = test.drop(["variety"], axis=1)
    train_y = train[["variety"]]
    test_y = test[["variety"]]

    # Hyperparameters
    penality = args.penality
    C = args.C

    experiment = mlflow.set_experiment(
        experiment_name="Classifier exp"
    )

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))

    with mlflow.start_run(run_name="Runer6", experiment_id=experiment.experiment_id):
        tags = {
            "engineering": "ML platform",
            "release.candidate": "RC1",
            "release.version": "2.0"
        }
        #set tags
        mlflow.set_tags(tags)

        # Model
        lr = LogisticRegression(penalty=penality, C=C)
        lr.fit(train_x, train_y)

        predicted_classes = lr.predict(test_x)

        (accuracy, precision, recall, f1) = eval_metrics(test_y, predicted_classes)

        print(f"Logistic Regression model (penality={penality}, C={C}):")
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1 Score: %s" % f1)

        # Logging parameters
        params = {
            "penality": penality,
            "C": C
        }
        mlflow.log_params(params)

        # Logging Metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Logging artifacts the data
        mlflow.log_artifacts("data/")
        # Logging model
        mlflow.sklearn.log_model(lr, "model")

        run = mlflow.last_active_run()
        print("Active run_id: {}".format(run.info.run_id))
