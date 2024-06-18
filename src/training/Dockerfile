# Dockerfile for training the model

# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_training.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements_training.txt

# Copy the rest of the application code into the container
COPY . .

# Run the training script
CMD ["python", "train_models.py"]
