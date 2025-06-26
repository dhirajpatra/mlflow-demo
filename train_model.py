import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
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
    wine_data = load_wine()

    X = wine_data.data
    y = wine_data.target

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
        # mlflow.log_param("n_estimators", n_estimators)
        # mlflow.log_param("max_depth", max_depth)
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("precision", precision)
        # mlflow.log_metric("recall", recall)
        # mlflow.log_metric("f1_score", f1)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Log the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="wine_quality_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )
        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Basic RM model for wine quality data"}
        )

        # Set tags for the model
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.set_tag("model_type", "RandomForestClassifier")
        logger.info("Model logged to MLflow Model Registry as 'WineQualityModel'.")

        # inferences
        load_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = load_model.predict(X_test)
        logger.info(f"Inference on test samples: {predictions}")

        wine_feature_names = wine_data.feature_names
        logger.info(f"Feature names: {wine_feature_names}")

        result_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions
        })
        logger.info("First 5 predictions:")
        logger.info(result_df)

        result = pd.DataFrame(X_test, columns=wine_feature_names)
        result['Actual'] = y_test
        result['Predicted'] = predictions
        logger.info(f"result DataFrame:\n{result.head().to_string()}")


