import os
import joblib
import json
import pandas as pd

def model_fn(model_dir):
    """
    1. Load the model artifact from the container directory.
    This runs once when the endpoint container boots up.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """
    2. Deserialize the incoming payload request.
    This safely parses standard application/json requests.
    """
    if request_content_type == "application/json":
        # Expecting JSON format: {"instances": [[5.1, 3.5, 1.4, 0.2]]}
        input_data = json.loads(request_body)
        dataset = pd.DataFrame(input_data["instances"])
        return dataset
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    3. Make predictions using the loaded model.
    """
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    """
    4. Serialize the prediction output back to the client.
    """
    if response_content_type == "application/json":
        # Convert numpy array to standard list for JSON serialization
        response = {"predictions": prediction.tolist()}
        return json.dumps(response), response_content_type
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
