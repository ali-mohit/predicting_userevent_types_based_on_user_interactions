# Predicting User Event Types Based on User Interactions with Products

## Overview

This project focuses on developing a machine learning model to predict user event types based on their interactions with products in a large multi-category online store. The dataset comprises seven months of behavior data, with each row representing an event linked to specific products and users. Events include actions such as viewing a product, adding it to the cart, and purchasing it.

## Authors

- Amin Aghasi
- Ali Mohit

## Proposal

### Problem Definition

The core challenge is to develop a machine learning model that can accurately classify user interactions into distinct event types. 

### Chosen Design Patterns

#### Data Representation Design Pattern

1. **Hashed Feature**:
   - **Category ID**: Use a hashed feature representation to handle the high cardinality of the `category_id`. This reduces dimensionality and handles missing values in `category_code`.

2. **Embeddings**:
   - **Brand**: Utilize embeddings for the `brand` feature to capture semantic similarities and user preferences. This simplifies the feature space and helps in understanding brand-related user behaviors.

3. **Feature Cross**:
   - **Temporal Features**: Extract features such as day of week and hour of day from the `event_time` column. Create feature crosses to capture variations in user behavior based on time.

#### Problem Representation Design Pattern

1. **Reframing**:
   - **Task Definition**: Reframe the problem as a multi-class classification task to predict user interactions (view, cart, purchase). This allows the use of classification algorithms to distinguish between different user event types.

2. **Rebalancing**:
   - **Data Imbalance**: Address the imbalance in the dataset (more views than purchases or cart events) by applying techniques such as oversampling the minority classes or undersampling the majority class. This ensures the model does not bias towards the majority class.

#### Model Representation Design Pattern

1. **Checkpoints**:
   - **Training Efficiency**: Implement checkpoints during model training to save the state at regular intervals. This allows resuming training from the last saved state in case of interruptions and helps in selecting the best model based on validation metrics.

2. **Hyperparameter Tuning**:
   - **Optimization**: Use hyperparameter tuning to find the best model configuration. This can be achieved through grid search or manual tuning to optimize parameters such as learning rate, number of epochs, and model architecture.

#### Serving Representation Design Pattern

1. **Stateless Serving Function**:
   - **Scalability**: Deploy the model using stateless serving functions to handle unpredictable workloads efficiently. Stateless functions are scalable and provide faster response times, which are crucial for real-time predictions.

## Implementation Steps

### Data Preprocessing

1. Handle missing values and inconsistencies in the dataset.
2. Apply hashed feature representation for `category_id`.
3. Generate embeddings for the `brand` feature.
4. Extract and cross temporal features from `event_time`.

### Model Training

1. Split the data into training, validation, and test sets.
2. Train a multi-class classification model using algorithms like Random Forest, SVM, or deep learning models.
3. Implement checkpoints and perform hyperparameter tuning to optimize the model.

### Evaluation

1. Evaluate the model using metrics such as accuracy, precision, recall, and F1-score for each class.
2. Use a confusion matrix to understand misclassifications and refine the model.

### Deployment

1. Deploy the trained model using stateless serving functions for real-time predictions.
2. Monitor the model's performance and update it periodically with new data.

## Project Structure
```
.
├── colab_results              # Directory for Colab results
│   ├── colab.py
│   └── README.md
├── k8s                        # Kubernetes manifests
│   ├── production
│   │   ├── deployment.yaml
│   │   ├── service-monitor.yaml
│   │   └── service.yaml
│   └── training
│       ├── cronjob.yaml
│       ├── service-monitor.yaml
│       └── service.yaml
├── src                        # Source code directory
│   ├── production
│   │   ├── app.py
│   │   ├── docker-compose.yaml
│   │   ├── Dockerfile
│   │   ├── prometheus.yml
│   │   └── requirements_production.txt
│   └── training
│       ├── docker-compose.yaml
│       ├── Dockerfile
│       ├── requirements_training.txt
│       └── train_models.py
└── README.md                  # Project README file
```

## Usage

### Running the Application

1. **Docker Compose**:
   ```sh
   docker-compose up -d
   ```
2. **Kubernetes: (Apply the manifests:)**:
   ```sh
    kubectl apply -f k8s/production/deployment.yaml
    kubectl apply -f k8s/production/service.yaml
    kubectl apply -f k8s/production/service-monitor.yaml
   ```

3. **CronJob for Training:**:
```sh
kubectl apply -f k8s/training/cronjob.yaml
```

## Accessing Metrics
### Prometheus:
* Visit http://localhost:9090 to access Prometheus UI and monitor metrics.

## API Endpoints
### Predict
* Endpoint: /predict
* Method: POST
* Payload: {"features": [...]}

## Conclusion

By following the outlined data, problem, model, and serving representation design patterns, we can effectively address the problem of predicting user event types based on user interactions with products. This structured approach ensures efficient handling of the dataset's complexities and enhances the model's performance and scalability.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries or support, please contact Ali Mohit.

