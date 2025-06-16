# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from preprocess import preprocess_data

# Define the registered model name
REGISTERED_MODEL_NAME = "RandomForestIrisClassifier"

def train():
    """
    Orchestrates the data preprocessing, model training,
    and comprehensive MLflow logging for a RandomForestClassifier.
    """
    print("Starting model training and MLflow logging...")

    # Set the MLflow experiment name
    mlflow.set_experiment("Iris RandomForest Classification")


    # 1. Data Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data()

    if X_train is None:
        print("Data preprocessing failed. Exiting training.")
        return

    with mlflow.start_run(run_name="RandomForest_Iris_Classification") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        client = MlflowClient() # Initialize MlflowClient inside the run or before

        # 2. Define Model Parameters
        n_estimators = 100
        max_depth = None
        random_state = 42

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_split_ratio", 0.2)

        # 3. Model Training
        print(f"Training RandomForestClassifier with n_estimators={n_estimators}, max_depth={max_depth}...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # 4. Model Evaluation
        print("Evaluating model performance...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"Logged Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

        # 5. Model Logging AND Registration
        print("Logging model to MLflow...")
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(1)

        # This call will log the model and register it
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_rf_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME
        )
        print(f"Model logged as artifact 'iris_rf_model' under run {run_id}.")
        print(f"Model also registered as '{REGISTERED_MODEL_NAME}'.")

        # 5a. Transition registered model to Production stage
        # Explicitly fetch the latest version of the registered model using MlflowClient
        try:
            # Get all versions of the model, sorted by creation timestamp to get the latest
            all_versions = client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'",
                order_by=["creation_timestamp DESC"]
            )
            if all_versions:
                latest_version_obj = all_versions[0]
                registered_model_version = latest_version_obj.version # This `version` attribute is on ModelVersion, not ModelInfo
                print(f"Found latest version {registered_model_version} (current stage: {latest_version_obj.current_stage}) for registered model {REGISTERED_MODEL_NAME}.")

                # Only transition if it's not already in Production
                if latest_version_obj.current_stage != "Production":
                    print(f"Transitioning model version {registered_model_version} of {REGISTERED_MODEL_NAME} from '{latest_version_obj.current_stage}' to 'Production' stage.")
                    client.transition_model_version_stage(
                        name=REGISTERED_MODEL_NAME,
                        version=registered_model_version,
                        stage="Production"
                    )
                    print(f"Model version {registered_model_version} of {REGISTERED_MODEL_NAME} successfully transitioned to 'Production' stage.")
                else:
                    print(f"Model version {registered_model_version} of {REGISTERED_MODEL_NAME} is already in 'Production' stage. No transition needed.")
            else:
                print(f"Warning: No versions found for registered model '{REGISTERED_MODEL_NAME}'. Cannot transition to Production.")
        except Exception as e:
            print(f"Error transitioning model to Production stage: {e}")

        # 6. Log additional artifacts
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame(
                {'feature': X_train.columns, 'importance': model.feature_importances_}
            ).sort_values('importance', ascending=False)
            feature_importances_path = "feature_importances.csv"
            feature_importances.to_csv(feature_importances_path, index=False)
            mlflow.log_artifact(feature_importances_path)
            print("Logged feature_importances.csv as an artifact.")

    print("MLflow run completed successfully!")

if __name__ == "__main__":
    train()