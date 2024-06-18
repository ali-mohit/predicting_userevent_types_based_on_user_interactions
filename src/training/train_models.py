import os
import zipfile
import requests
import pandas as pd
from dask import dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import boto3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from prometheus_client import start_http_server, Summary, Gauge

# Create Prometheus metrics
accuracy_metric = Gauge('model_accuracy', 'Accuracy of the model', ['model'])
f1_metric = Gauge('model_f1_score', 'F1 Score of the model', ['model'])
training_time = Summary('model_training_duration_seconds', 'Time spent training model')

# Environment variables
dataset_url = os.environ['DATASET_URL']
s3_host = os.environ['S3_HOST']
s3_username = os.environ['S3_USERNAME']
s3_password = os.environ['S3_PASSWORD']

# Start Prometheus server to expose metrics
start_http_server(8001)

# Download and extract the datasets
response = requests.get(dataset_url)
zip_file_path = '/app/datasets.zip'
with open(zip_file_path, 'wb') as f:
    f.write(response.content)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('/app/datasets')

# List of extracted files
file_path = '/app/datasets'
file_list = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.csv')]

# Load and combine datasets using Dask
ddf = dd.read_csv(file_list)

# Preprocess the data
# Handle missing values
ddf = ddf.fillna({'category_code': 'Unknown', 'brand': 'Unknown', 'user_session': 'Unknown'})

# Convert event_time to datetime and extract temporal features
ddf['event_time'] = dd.to_datetime(ddf['event_time'])
ddf['day_of_week'] = ddf['event_time'].dt.dayofweek
ddf['hour_of_day'] = ddf['event_time'].dt.hour

# Generate embeddings for brand feature using LabelEncoder
ddf['brand_encoded'] = dd.to_numeric(ddf['brand'], errors='coerce').fillna(-1).astype(int)

# Prepare the feature set
features = ddf[['price', 'day_of_week', 'hour_of_day', 'brand_encoded']]
target = ddf['event_type']

# Convert target to numerical labels
target = target.map({'view': 0, 'cart': 1, 'purchase': 2}).astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=True)

# Standardize the features using Dask
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compute the scaled arrays only once before applying SMOTE
X_train_np = X_train.compute()
y_train_np = y_train.compute()

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_np, y_train_np)

# Save function for model checkpoints
def save_checkpoint(model, filename):
    joblib.dump(model, filename)

@training_time.time()
def train_and_evaluate():
    # Train and checkpoint Logistic Regression model using SGDClassifier
    logistic_model = SGDClassifier(max_iter=1000, tol=1e-3)
    logistic_model.partial_fit(X_resampled, y_resampled, classes=np.unique(y_resampled))
    save_checkpoint(logistic_model, '/app/models/logistic_model_checkpoint.pkl')

    # Train and checkpoint Neural Network model using MLPClassifier
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    nn_model.fit(X_resampled, y_resampled)
    save_checkpoint(nn_model, '/app/models/nn_model_checkpoint.pkl')

    # Train and checkpoint Deep Learning model using TensorFlow/Keras
    y_resampled_categorical = to_categorical(y_resampled)
    y_test_categorical = to_categorical(y_test.compute())

    dl_model = Sequential([
        Dense(128, input_dim=X_resampled.shape[1], activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint('/app/models/dl_model_checkpoint.h5', save_best_only=True)
    dl_model.fit(X_resampled, y_resampled_categorical, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint_cb])

    # Evaluate the models and expose metrics
    y_pred_logistic = logistic_model.predict(X_test.compute())
    logistic_accuracy = accuracy_score(y_test.compute(), y_pred_logistic)
    logistic_f1 = f1_score(y_test.compute(), y_pred_logistic, average='weighted')
    accuracy_metric.labels('logistic_regression').set(logistic_accuracy)
    f1_metric.labels('logistic_regression').set(logistic_f1)

    y_pred_nn = nn_model.predict(X_test.compute())
    nn_accuracy = accuracy_score(y_test.compute(), y_pred_nn)
    nn_f1 = f1_score(y_test.compute(), y_pred_nn, average='weighted')
    accuracy_metric.labels('neural_network').set(nn_accuracy)
    f1_metric.labels('neural_network').set(nn_f1)

    dl_model.load_weights('/app/models/dl_model_checkpoint.h5')
    y_pred_dl = dl_model.predict(X_test.compute())
    y_pred_dl_classes = y_pred_dl.argmax(axis=1)
    dl_accuracy = accuracy_score(y_test.compute(), y_pred_dl_classes)
    dl_f1 = f1_score(y_test.compute(), y_pred_dl_classes, average='weighted')
    accuracy_metric.labels('deep_learning').set(dl_accuracy)
    f1_metric.labels('deep_learning').set(dl_f1)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_resampled, y_resampled)
    save_checkpoint(rf_model, '/app/models/rf_model_checkpoint.pkl')

    y_pred_rf = rf_model.predict(X_test.compute())
    rf_accuracy = accuracy_score(y_test.compute(), y_pred_rf)
    rf_f1 = f1_score(y_test.compute(), y_pred_rf, average='weighted')
    accuracy_metric.labels('random_forest').set(rf_accuracy)
    f1_metric.labels('random_forest').set(rf_f1)

    # Upload models to S3
    s3_client = boto3.client('s3', endpoint_url=s3_host, aws_access_key_id=s3_username, aws_secret_access_key=s3_password)
    bucket_name = os.environ('S3_PRODUCTION_BUCKET', 'our-bucket-name')

    models = [
        ('/app/models/logistic_model_checkpoint.pkl', 'logistic_model_checkpoint.pkl'),
        ('/app/models/nn_model_checkpoint.pkl', 'nn_model_checkpoint.pkl'),
        ('/app/models/dl_model_checkpoint.h5', 'dl_model_checkpoint.h5'),
        ('/app/models/rf_model_checkpoint.pkl', 'rf_model_checkpoint.pkl')
    ]

    for model_path, model_key in models:
        s3_client.upload_file(model_path, bucket_name, model_key)

if __name__ == "__main__":
    train_and_evaluate()
