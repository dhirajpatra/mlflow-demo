# evaluate.py
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_data # Assuming preprocess_data is the function from preprocess.py
import sys
import os

def evaluate_model(run_id_or_model_uri=None):
    """
    Loads a trained model from MLflow, preprocesses data,
    and evaluates the model's performance on the test set.

    Args:
        run_id_or_model_uri (str, optional): The MLflow Run ID (e.g., "1a2b3c4d5e")
                                             or the full MLflow Model URI
                                             (e.g., "mlruns:/1a2b3c4d5e/iris_rf_model"
                                             or "models:/RandomForestIrisClassifier/Production").
                                             If None, it tries to load the latest
                                             model from the "RandomForestIrisClassifier" registered model.
    """
    print("Starting model evaluation...")

    # 1. Data Preprocessing
    # Ensure preprocess_data is imported correctly
    X_train, X_test, y_train, y_test = preprocess_data()

    if X_test is None or y_test is None:
        print("Data preprocessing failed or returned None. Exiting evaluation.")
        return

    print(f"Test set shape: {X_test.shape}")

    # 2. Load Model from MLflow
    model = None
    if run_id_or_model_uri:
        model_uri = run_id_or_model_uri
        print(f"\nAttempting to load model from: {model_uri}")
    else:
        # Default to loading the latest 'Production' stage model if no specific URI/ID is provided
        model_uri = sys.argv[1] if len(sys.argv) > 1 else "models:/RandomForestIrisClassifier@Production" # Use alias syntax
        print(f"\nNo specific RUN_ID or Model URI provided. Attempting to load latest 'Production' model from: {model_uri}")
        # Alternatively, you could try to find the latest run if you don't use registered models often:
        # runs = mlflow.search_runs(order_by=["metrics.accuracy DESC"], max_results=1)
        # if not runs.empty:
        #     latest_run_id = runs.iloc[0].run_id
        #     model_uri = f"mlruns:/{latest_run_id}/iris_rf_model"
        #     print(f"Found latest run {latest_run_id}. Loading model from: {model_uri}")
        # else:
        #     print("Could not find any MLflow runs. Please specify a RUN_ID or Model URI.")
        #     sys.exit(1)

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from '{model_uri}': {e}")
        print("Please ensure the RUN_ID or Model URI is correct and the model artifact exists.")
        sys.exit(1) # Exit if model cannot be loaded

    # 3. Make Predictions
    print("\nMaking predictions on the test set...")
    try:
        y_pred = model.predict(X_test)
        print("Predictions generated.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

    # 4. Generate and Print Evaluation Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 5. Log Evaluation Metrics (optional, but good for tracking evaluation runs)
    # You could create a new MLflow run specifically for this evaluation,
    # or you could just print. Logging provides a record.
    # Let's create a new run to log evaluation results.
    print("\nLogging evaluation metrics to a new MLflow run...")
    with mlflow.start_run(run_name="Model_Evaluation_Run") as run:
        mlflow.log_param("evaluated_model_uri", model_uri)

        # Log individual metrics for easier comparison in MLflow UI
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric("eval_accuracy", accuracy)
        mlflow.log_metric("eval_precision", precision)
        mlflow.log_metric("eval_recall", recall)
        mlflow.log_metric("eval_f1_score", f1)

        print(f"Evaluation metrics logged under MLflow Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    print("\nModel evaluation complete!")

if __name__ == "__main__":
    # --- How to run this script ---
    # Option 1: Evaluate the latest 'Production' stage of the registered model
    evaluate_model()

    # Option 2: Evaluate a model from a specific MLflow Run ID
    # Replace "<YOUR_TRAINING_RUN_ID>" with an actual ID from your MLflow UI
    # For example: evaluate_model(run_id_or_model_uri="runs:/abcdef1234567890/iris_rf_model")

    # Option 3: Evaluate a specific version of a registered model
    # For example: evaluate_model(run_id_or_model_uri="models:/RandomForestIrisClassifier/2")