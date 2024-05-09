# Use a Python base image
FROM python:3.9-slim

# Install system dependencies (for CUDA and other tools)
RUN apt-get update && apt-get install -y \
    libgl1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install flask diffusers transformers invisible_watermark accelerate safetensors torch torchvision

# Set the working directory
WORKDIR /app

# Copy the Flask application into the Docker image
COPY . /app

# Define the command to start the Flask server
CMD ["python", "app.py"]
