from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import joblib
import numpy as np
import boto3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Create a custom metric to track model predictions
model_predicts = metrics.counter(
    'model_predictions', 'Number of model predictions',
    labels={'model': lambda: request.args.get('model', 'unknown')}
)

# Environment variables
s3_host = os.environ['S3_HOST']
s3_username = os.environ['S3_USERNAME']
s3_password = os.environ['S3_PASSWORD']
bucket_name = os.environ('S3_PRODUCTION_BUCKET', 'our-bucket-name')

# S3 client
s3_client = boto3.client('s3', endpoint_url=s3_host, aws_access_key_id=s3_username, aws_secret_access_key=s3_password)

# Download models from S3
models = [
    ('logistic_model_checkpoint.pkl', 'logistic_model_checkpoint.pkl'),
    ('nn_model_checkpoint.pkl', 'nn_model_checkpoint.pkl'),
    ('dl_model_checkpoint.h5', 'dl_model_checkpoint.h5'),
    ('rf_model_checkpoint.pkl', 'rf_model_checkpoint.pkl')
]

for model_key, model_filename in models:
    s3_client.download_file(bucket_name, model_key, f'/app/models/{model_filename}')

# Load the models
logistic_model = joblib.load('/app/models/logistic_model_checkpoint.pkl')
nn_model = joblib.load('/app/models/nn_model_checkpoint.pkl')
rf_model = joblib.load('/app/models/rf_model_checkpoint.pkl')
dl_model = load_model('/app/models/dl_model_checkpoint.h5')

@app.route('/predict', methods=['POST'])
@model_predicts.count_exceptions()
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    
    model = request.args.get('model', 'all')

    predictions = {}
    if model in ['logistic', 'all']:
        logistic_pred = logistic_model.predict(features)
        predictions['logistic_prediction'] = int(logistic_pred[0])
    if model in ['nn', 'all']:
        nn_pred = nn_model.predict(features)
        predictions['nn_prediction'] = int(nn_pred[0])
    if model in ['rf', 'all']:
        rf_pred = rf_model.predict(features)
        predictions['rf_prediction'] = int(rf_pred[0])
    if model in ['dl', 'all']:
        dl_pred = dl_model.predict(features)
        dl_pred_class = np.argmax(dl_pred, axis=1)
        predictions['dl_prediction'] = int(dl_pred_class[0])
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
