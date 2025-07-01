import os
import mlflow
import pandas as pd
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model_name = "tracking-quickstart"  # Name of the model registered in MLflow
model_version = "5" # or Staging, depending on your deployment

try:
    # load the model
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"  # Adjust the URI
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    logger.info(f"Model {model_name} version {model_version} loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model {model_name} version {model_version}: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    {
        "predictions": [
            2
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:    
        data = request.get_json(force=True)

        # Example input data for testing
        # data = [
        #     [14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]
        #     ]
        # input_data = {
        #     "fixed acidity": 14.34,
        #     "volatile acidity": 1.68,
        #     "citric acid": 2.7,
        #     "residual sugar": 25.0,
        #     "chlorides": 98.0,
        #     "free sulfur dioxide": 2.8,
        #     "total sulfur dioxide": 1.31,
        #     "density": 0.53,
        #     "pH": 2.7,
        #     "sulphates": 13.0,
        #     "alcohol": 0.57,
        #     "magnesium": 1.96,
        #     "ash": 660.0
        # }


        # Get the input data from the request
        logger.info(f"Received data for prediction: {data}")

        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"}), 400

        # Make predictions
        predictions = model.predict(input_df)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def index():
    return "Welcome to the Wine Quality Classification Model API!"

if __name__ == '__main__':
    app.run(host='0.0.0', port=5001, debug=True) # for local testing, use port 5001

