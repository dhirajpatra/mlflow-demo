import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Function to evaluate model metrics
def evaluate_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted', zero_division=0)
    recall = recall_score(actual, pred, average='weighted', zero_division=0)
    f1 = f1_score(actual, pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Load data (using a simple dummy dataset for demonstration)
    # In a real scenario, this would involve more robust data loading/preprocessing
    data = {
        'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 7.3, 7.8, 7.5],
        'volatile acidity': [0.70, 0.88, 0.76, 0.28, 0.70, 0.66, 0.60, 0.65, 0.58, 0.50],
        'citric acid': [0.00, 0.00, 0.04, 0.56, 0.00, 0.00, 0.06, 0.00, 0.02, 0.36],
        'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.2, 2.0, 6.1],
        'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.069, 0.065, 0.073, 0.071],
        'free sulfur dioxide': [11.0, 25.0, 40.0, 17.0, 11.0, 13.0, 15.0, 15.0, 9.0, 10.0],
        'total sulfur dioxide': [34.0, 67.0, 60.0, 60.0, 34.0, 40.0, 59.0, 21.0, 18.0, 24.0],
        'density': [0.9978, 0.9968, 0.9968, 0.9980, 0.9978, 0.9978, 0.9964, 0.9946, 0.9968, 0.9978],
        'pH': [3.51, 3.20, 3.26, 3.16, 3.51, 3.51, 3.30, 3.39, 3.36, 3.38],
        'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56, 0.56, 0.46, 0.47, 0.57, 0.58],
        'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 10.8, 10.0, 9.5, 9.7],
        'quality': [5, 5, 5, 6, 5, 5, 5, 7, 7, 5] # Target variable
    }
    wine_df = pd.DataFrame(data)

    X = wine_df.drop("quality", axis=1)
    y = wine_df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    n_estimators = 100
    max_depth = 10

    # Start MLflow run
    mlflow.set_tracking_uri("http://localhost:5000")  # Adjust the URI
    mlflow.set_experiment("WineQuality_Classification")

    with mlflow.start_run(run_name="RandomForest_WineQuality"):
        logger.info("Starting training of RandomForestClassifier...")
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model
        accuracy, precision, recall, f1 = evaluate_metrics(y_test, y_pred)

        logger.info(f"RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth}):")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model (Scikit-learn flavor)
        mlflow.sklearn.log_model(model, "wine_quality_model", registered_model_name="WineQualityModel")
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.set_tag("model_type", "RandomForestClassifier")
        logger.info("Model logged to MLflow Model Registry as 'WineQualityModel'.")
