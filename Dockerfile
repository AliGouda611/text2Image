# Use a Python base image
FROM python:3.9

# Install required Python packages
RUN pip install flask diffusers transformers invisible_watermark accelerate safetensors torch torchvision

# Copy the Flask app into the Docker image
COPY . /app

# Set the working directory
WORKDIR /app

# Run the Flask server when the container starts
CMD ["python", "app.py"]
