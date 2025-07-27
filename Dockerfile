# Step 1: Specify the base image and platform
FROM --platform=linux/amd64 python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy dependency files
COPY requirements.txt .

# Step 4: Install system dependencies (for PyMuPDF) and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Download and cache models to ensure offline execution
COPY download_models.py .
RUN python download_models.py

# Step 6: Copy your application source code
COPY ./src ./src

# Step 7: Define the command to run your application
CMD ["python", "src/main.py"]