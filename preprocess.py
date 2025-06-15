import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def preprocess():
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test
