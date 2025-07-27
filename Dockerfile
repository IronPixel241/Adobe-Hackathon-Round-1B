# Step 1: Specify the base image and platform
FROM --platform=linux/amd64 python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Install dependencies (needs internet)
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Download and cache models (needs internet)
COPY download_models.py .
RUN python download_models.py

# Step 5: NOW, set the environment to offline for the final container runtime
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Step 6: Copy your application source code
COPY ./src ./src

# Step 7: Define the command to run your application
CMD ["python", "src/main.py"]