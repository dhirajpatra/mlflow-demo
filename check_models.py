# check_models.py
import mlflow
from mlflow.tracking import MlflowClient
import datetime

def format_timestamp(ts):
    """Helper to format Unix timestamps to human-readable strings."""
    if ts is None:
        return "N/A"
    return datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

def check_registered_models():
    """
    Connects to MLflow Tracking Server and lists all registered models
    and their versions, along with their current stages.
    """
    client = MlflowClient()

    print(f"Connecting to MLflow Tracking URI: {mlflow.get_tracking_uri()}\n")
    print("--- Listing All Registered Models and Versions ---")

    try:
        models = client.search_registered_models()
        if not models:
            print("No registered models found.")
            return

        for model in models:
            print(f"\nModel Name: {model.name}")
            print(f"  Description: {model.description if model.description else 'N/A'}")
            print(f"  Creation Time: {format_timestamp(model.creation_timestamp)}")
            print(f"  Last Updated: {format_timestamp(model.last_updated_timestamp)}")
            print(f"  Tags: {model.tags}")

            # Fetch all versions for this model name, sorted by version number
            versions = client.search_model_versions(f"name='{model.name}'")
            if not versions:
                print("    No versions found for this model.")
                continue

            for version in versions:
                print(f"    --- Version: {version.version} ---")
                print(f"      Stage: {version.current_stage}")
                print(f"      Run ID: {version.run_id}")
                print(f"      Source: {version.source}")
                print(f"      Creation Time: {format_timestamp(version.creation_timestamp)}")
                print(f"      Last Updated: {format_timestamp(version.last_updated_timestamp)}")
                print(f"      Status: {version.status}")
                print(f"      Status Message: {version.status_message if version.status_message else 'N/A'}")
                print(f"      Tags: {version.tags}")


    except Exception as e:
        print(f"An error occurred while listing models: {e}")
        print("Please ensure your MLflow Tracking Server is running and accessible (if remote).")
        print("For local file storage, check file permissions in your 'mlruns' directory.")

if __name__ == "__main__":
    check_registered_models()