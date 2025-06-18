import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

# Create a directory to store downloaded files if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download red-wine-quality.csv
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red_wine_file_path = "data/red-wine-quality.csv"

try:
    red_wine_data = pd.read_csv(red_wine_url, sep=';')
    red_wine_data.to_csv(red_wine_file_path, index=False)
    print(f"Downloaded red-wine-quality.csv to {red_wine_file_path}")
except Exception as e:
    print(f"Error downloading red-wine-quality.csv: {e}")

# MLflow Program for Iris Dataset
# Set MLflow tracking URI (optional, can be a local directory or a remote server)
# mlflow.set_tracking_uri("sqlite:///mlruns.db") # For local database

# Set experiment name
mlflow.set_experiment("Iris Classification Experiment")

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for the full dataset to log as artifact
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['variety'] = [target_names[i] for i in y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "LogisticRegression")

    # Train the model
    model = LogisticRegression(max_iter=200) # Increased max_iter for convergence
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log artifacts
    # Log the full dataset
    iris_df.to_csv("iris_full_dataset.csv", index=False)
    mlflow.log_artifact("iris_full_dataset.csv", "data")

    # Log train and test data (as separate CSVs for simplicity)
    pd.DataFrame(X_train, columns=feature_names).to_csv("iris_train_features.csv", index=False)
    pd.DataFrame(y_train, columns=['variety_id']).to_csv("iris_train_target.csv", index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv("iris_test_features.csv", index=False)
    pd.DataFrame(y_test, columns=['variety_id']).to_csv("iris_test_target.csv", index=False)

    mlflow.log_artifact("iris_train_features.csv", "data/train")
    mlflow.log_artifact("iris_train_target.csv", "data/train")
    mlflow.log_artifact("iris_test_features.csv", "data/test")
    mlflow.log_artifact("iris_test_target.csv", "data/test")

    # Log the model
    mlflow.sklearn.log_model(model, "iris_model")

    # Set multiple tags
    mlflow.set_tag("model_family", "Classification")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("purpose", "Petal Variety Classification")

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Print the last active run
last_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
if not last_run.empty:
    print(last_run)  # This line prints the full DataFrame of the last run
    print("\n--- Last Active MLflow Run Details ---")
    print(f"Run ID: {last_run.iloc[0].run_id}")
    print(f"Experiment ID: {last_run.iloc[0].experiment_id}")
    print(f"Status: {last_run.iloc[0].status}")
    print(f"Start Time: {last_run.iloc[0].start_time}")
    print(f"End Time: {last_run.iloc[0].end_time}")

    # Corrected way to print tags and metrics from flattened columns
    print("Tags:")
    for col in last_run.columns:
        if col.startswith('tags.'):
            tag_name = col[len('tags.'):]
            print(f"  {tag_name}: {last_run.iloc[0][col]}")

    print("Metrics:")
    for col in last_run.columns:
        if col.startswith('metrics.'):
            metric_name = col[len('metrics.'):]
            print(f"  {metric_name}: {last_run.iloc[0][col]}")
else:
    print("\nNo MLflow runs found.")