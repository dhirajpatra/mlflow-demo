#!/bin/bash

# run_ml_pipeline.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# remove the existing mlruns directory if it exists
echo -e "\n--- Cleaning up previous MLflow runs directory (if exists) ---"
rm -rf mlruns

# --- Configuration ---
# Set the desired MLflow experiment name
MLFLOW_EXPERIMENT_NAME="Iris RandomForest Classification"

# Define the model name used in train.py for registration
REGISTERED_MODEL_NAME="RandomForestIrisClassifier"

# Define the artifact path used in train.py for logging the model
MODEL_ARTIFACT_PATH="iris_rf_model"

# --- Colors for better output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}--- Starting ML Pipeline Automation Script ---${NC}"

# --- Step 1: Pre-run Checks ---
echo -e "${YELLOW}\n--- Performing Pre-run Checks ---${NC}"

# Check for Python executable
if ! command -v python3 &> /dev/null
then
    echo -e "${RED}Error: python3 command not found. Please ensure Python 3 is installed and in your PATH.${NC}"
    exit 1
fi

# Check for virtual environment activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated. It's highly recommended to activate your 'myenv' virtual environment first.${NC}"
    echo -e "${YELLOW}You can activate it using: source /path/to/myenv/bin/activate${NC}"
fi

# Check if required Python scripts exist
if [ ! -f "preprocess.py" ]; then
    echo -e "${RED}Error: preprocess.py not found in the current directory.${NC}"
    exit 1
fi
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found in the current directory.${NC}"
    exit 1
fi
if [ ! -f "evaluate.py" ]; then
    echo -e "${RED}Error: evaluate.py not found in the current directory.${NC}"
    exit 1
fi

echo -e "${GREEN}Pre-run checks passed.${NC}"

# --- Step 2: Clean MLflow Tracking Directory (Optional, but good for fresh runs) ---
echo -e "${YELLOW}\n--- Cleaning MLflow Tracking Directory ---${NC}"
if [ -d "mlruns" ]; then
    echo -e "Found 'mlruns' directory. Deleting it to ensure a clean slate."
    rm -rf mlruns
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}'mlruns' directory deleted successfully.${NC}"
    else
        echo -e "${RED}Error: Failed to delete 'mlruns' directory. Check permissions.${NC}"
        exit 1
    fi
else
    echo -e "No 'mlruns' directory found. Starting fresh."
fi

# --- Step 3: Set MLflow Environment Variables ---
export MLFLOW_TRACKING_URI="file:./mlruns"
echo -e "${BLUE}MLFLOW_TRACKING_URI set to: ${MLFLOW_TRACKING_URI}${NC}"

# --- Step 4: Run Training Script ---
echo -e "${YELLOW}\n--- Running Training Script (train.py) ---${NC}"
python3 train.py
TRAIN_STATUS=$? # Capture the exit status of the previous command

if [ $TRAIN_STATUS -eq 0 ]; then
    echo -e "${GREEN}Training script completed successfully.${NC}"
else
    echo -e "${RED}Error: Training script failed with exit code ${TRAIN_STATUS}. Please check the output above for details.${NC}"
    exit 1
fi

# --- Step 5: Extract Latest Run ID for Evaluation ---
echo -e "${YELLOW}\n--- Attempting to get the latest model URI for evaluation ---${NC}"

# Option 1: Load from Registered Model (Recommended if you're using model registry)
MODEL_URI="models:/${REGISTERED_MODEL_NAME}/Production" # Or /Staging, or /Latest
echo -e "${BLUE}Attempting to evaluate the model from Registered Model: ${MODEL_URI}${NC}"
# You could add logic here to check if the model actually exists in this stage

# Option 2 (Fallback/Alternative): Get latest run ID and construct URI (less robust if multiple experiments)
# This will find the latest run in the default experiment (or whatever is implicitly active)
# It's better to explicitly search for runs within the experiment you just created.
# You might need to adjust the experiment ID based on your mlflow.set_experiment() in train.py
# If you didn't explicitly set an experiment in train.py, it's ID 0 "Default"
# Otherwise, you need to search for the ID of "${MLFLOW_EXPERIMENT_NAME}"
# For simplicity and assuming you explicitly set the experiment name in train.py, let's try to get its ID
# Otherwise, we rely on the registered model.

# --- START OF MODIFICATION ---
# The following lines are commented out or modified because your MLflow CLI version
# does not support the '--filter' option for 'mlflow experiments search'.
# This ensures the script relies on the direct 'models:/<name>/Production' URI.

# EXPERIMENT_ID=$(mlflow experiments search --filter "name = '${MLFLOW_EXPERIMENT_NAME}'" --output-as-json | jq -r '.[0].experiment_id')
# if [ -n "$EXPERIMENT_ID" ]; then
#     echo -e "Found Experiment ID for '${MLFLOW_EXPERIMENT_NAME}': ${EXPERIMENT_ID}"
#     LATEST_RUN_ID=$(mlflow runs search --experiment-ids "$EXPERIMENT_ID" --order-by "start_time DESC" --max-results 1 --output-as-json | jq -r '.[0].run_id')

#     if [ -n "$LATEST_RUN_ID" ]; then
#         echo -e "${GREEN}Found latest run ID: ${LATEST_RUN_ID} in experiment ${EXPERIMENT_ID}.${NC}"
#         RUN_MODEL_URI="runs:/${LATEST_RUN_ID}/${MODEL_ARTIFACT_PATH}"
#         echo -e "${BLUE}Constructed Run Model URI: ${RUN_MODEL_URI}${NC}"
#     else
#         echo -e "${YELLOW}Warning: Could not find latest run ID in experiment ${EXPERIMENT_ID}. Proceeding with registered model URI.${NC}"
#     fi
# else
#     echo -e "${YELLOW}Warning: Could not find Experiment ID for '${MLFLOW_EXPERIMENT_NAME}'. Proceeding with registered model URI.${NC}"
# fi
# --- END OF MODIFICATION ---

# --- Step 6: Run Evaluation Script ---
echo -e "${YELLOW}\n--- Running Evaluation Script (evaluate.py) ---${NC}"
# Pass the model URI (registered model) to the evaluate script
# If you wanted to use the RUN_MODEL_URI from the previous step, uncomment this:
# python3 evaluate.py "$RUN_MODEL_URI"
python3 evaluate.py "$MODEL_URI"
EVAL_STATUS=$?

if [ $EVAL_STATUS -eq 0 ]; then
    echo -e "${GREEN}Evaluation script completed successfully.${NC}"
else
    echo -e "${RED}Error: Evaluation script failed with exit code ${EVAL_STATUS}. Please check the output above for details.${NC}"
    exit 1
fi

echo -e "${BLUE}\n--- ML Pipeline Automation Script Finished ---${NC}"
echo -e "${BLUE}You can view your MLflow runs by running: mlflow ui${NC}"